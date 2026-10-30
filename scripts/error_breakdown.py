"""Which of the MAP's errors does each score catch, and why?

On an evaluate_bio.py output (bio or PopQA corpus), for the "wrong" task
(seen probes, f >= 1):

  error types   the MAP's teacher-forced argmax answer is
                  swap   another training answer of the same attribute
                         (the fact confused with someone else's)
                  other  anything else
                and, separately, first answer token wrong vs only later ones
  train fit     mean MAP NLL of the answer tokens at the fact's occurrences
                in the training documents: how well the fine-tune fit it
  per type      within-entropy AUROC, errors of that type vs all correct
  per f         AUROC of errors vs correct within each training frequency
  logistic      held-out (folds by name): softmax, + train fit, + each score,
                + train fit + each score: does a score add beyond knowing how
                well the fact was fit?
  overlap       at the 20% most uncertain seen probes per score, the errors
                caught by both scores, only one, or neither

  uv run python scripts/error_breakdown.py --corpus data/bio_err1/corpus.json \\
      --adapter checkpoints/bio_err1_lora --eval figures/bio_eval_err1_ens16.json \\
      --out figures/error_breakdown_bio1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_posterior import stratified_auroc
from combine_bio import heldout_auroc, probe_ids
from evaluate_bio import encode_probes, spearman
from evaluate_posterior import auroc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True, help="evaluate_bio.py output for this corpus")
    ap.add_argument("--scores", nargs="+", default=[
        "post=token/projected_alternating25_measured MI", "ens=baseline/lora_ensemble MI",
        "1-conf=MAP 1-conf"], help="alias=score key; the first two are compared in 'overlap'")
    ap.add_argument("--top", type=float, default=0.2, help="share flagged in 'overlap'")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def next_token_logprobs(model, ids, att, device, batch_size):
    """log p(token_t | <t) for t >= 1 and the argmax prediction, per row."""
    lp, am = [], []
    for i in range(0, ids.shape[0], batch_size):
        x, a = ids[i:i + batch_size].to(device), att[i:i + batch_size].to(device)
        logits = model(input_ids=x, attention_mask=a).logits[:, :-1].float().log_softmax(-1)
        lp.append(logits.gather(-1, x[:, 1:, None])[..., 0].cpu())
        am.append(logits.argmax(-1).cpu())
    return torch.cat(lp), torch.cat(am)


def answer_spans(doc, q):
    """Character spans of the probe's answer in a training document."""
    ans, spans = q["answer"].strip(), []
    if q["prompt"] in doc:  # PopQA: "Q: ...\nA:" followed by " answer"
        start = 0
        while (j := doc.find(q["prompt"], start)) >= 0:
            k = j + len(q["prompt"]) + 1
            if doc[k:k + len(ans)] == ans:
                spans.append((k, k + len(ans)))
            start = j + 1
    elif q["name"] in doc:  # bio: one person per document, the value after the name
        k = doc.find(ans, doc.find(q["name"]))
        if k >= 0:
            spans.append((k, k + len(ans)))
    return spans


def main():
    args = parse_args()
    corpus = json.loads(Path(args.corpus).read_text())
    probes, docs = corpus["probes"], corpus["texts"]
    r = json.load(open(args.eval))["report"]
    freq, correct = np.array(r["probe_freq"]), np.array(r["probe_correct"], bool)
    if [q["freq"] for q in probes] != r["probe_freq"]:
        raise SystemExit("eval and corpus probes differ")
    scores = {a: torch.tensor(r["probe_scores"][k], dtype=torch.float64)
              for a, k in (s.split("=", 1) for s in args.scores)}
    pid = probe_ids({"probe_name": [q["name"] for q in probes], "probe_freq": r["probe_freq"]})

    tok = AutoTokenizer.from_pretrained(args.adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(args.base_model, attn_implementation="eager",
                                                dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()

    # the MAP's teacher-forced answer per probe
    enc = encode_probes(tok, probes)
    _, pred = next_token_logprobs(model, enc["input_ids"], enc["attention_mask"], args.device,
                                  args.batch_size)
    m = enc["loss_mask"][:, 1:].bool()
    pred_str, first_wrong = [], np.zeros(len(probes), bool)
    for i in range(len(probes)):
        p, t = pred[i][m[i]], enc["input_ids"][i, 1:][m[i]]
        pred_str.append(tok.decode(p))
        first_wrong[i] = bool(p[0] != t[0])
    values = {}
    for q in probes:
        if q["freq"] >= 1:
            values.setdefault(q["attr"], set()).add(q["answer"].strip())
    swap = np.array([s.strip() in values.get(q["attr"], ()) and s.strip() != q["answer"].strip()
                     for s, q in zip(pred_str, probes)])

    # how well the fine-tune fit each fact: answer-token NLL at its training occurrences
    denc = tok(docs, return_offsets_mapping=True, padding=True, return_tensors="pt")
    lp, _ = next_token_logprobs(model, denc["input_ids"], denc["attention_mask"], args.device,
                                args.batch_size)
    offs = denc["offset_mapping"]
    by_name = {}
    for d, doc in enumerate(docs):
        for i, q in enumerate(probes):
            if q["freq"] >= 1 and q["name"] in doc:
                by_name.setdefault(i, []).append(d)
    fit_nll = np.full(len(probes), np.nan)
    for i, ds in by_name.items():
        vals = []
        for d in ds:
            for a, b in answer_spans(docs[d], probes[i]):
                t = [j for j in range(1, offs.shape[1]) if offs[d, j, 0] < b and offs[d, j, 1] > a
                     and denc["attention_mask"][d, j]]
                if t:
                    vals.append(float(-lp[d, [j - 1 for j in t]].mean()))
        if vals:
            fit_nll[i] = np.mean(vals)
    seen = freq >= 1
    found = seen & ~np.isnan(fit_nll)
    print(f"train fit located for {found.sum()} / {seen.sum()} seen probes")
    scores["train fit"] = torch.from_numpy(np.nan_to_num(fit_nll, nan=np.nanmean(fit_nll)))

    out = {"eval": args.eval, "n_seen": int(seen.sum()), "n_wrong": int((seen & ~correct).sum())}
    strata = torch.tensor(r["probe_scores"]["MAP entropy"], dtype=torch.float64)

    def within(pos, neg, s):
        keep = torch.from_numpy(pos | neg)
        return stratified_auroc(torch.from_numpy(pos)[keep], s[keep], strata[keep], 5)["weighted"]

    wrong = seen & ~correct
    types = {"swap": wrong & swap, "other": wrong & ~swap,
             "first token wrong": wrong & first_wrong, "later token wrong": wrong & ~first_wrong}
    neg = seen & correct
    out["types"] = {}
    print(f"\n{wrong.sum()} errors among {seen.sum()} seen probes; within-entropy AUROC by type")
    print(f"  {'type':18s} {'n':>5s} " + " ".join(f"{a:>10s}" for a in scores))
    for name, pos in types.items():
        row = {"n": int(pos.sum()), **{a: within(pos, neg, s) if pos.sum() >= 5 else float("nan")
                                       for a, s in scores.items()}}
        out["types"][name] = row
        print(f"  {name:18s} {row['n']:5d} " + " ".join(f"{row[a]:10.3f}" for a in scores))

    out["per_freq"] = {}
    print("\nraw AUROC, errors vs correct within each training frequency")
    for f in sorted(set(freq[seen].tolist())):
        at = freq == f
        e, c = wrong & at, neg & at
        if e.sum() < 5 or c.sum() < 5:
            continue
        row = {"n_wrong": int(e.sum()), "n_right": int(c.sum()),
               **{a: auroc(s[torch.from_numpy(c)], s[torch.from_numpy(e)]) for a, s in scores.items()}}
        out["per_freq"][f] = row
        print(f"  f={f:<3d} {row['n_wrong']:4d}/{row['n_right']:<5d} "
              + " ".join(f"{a} {row[a]:.3f}" for a in scores))

    out["spearman_train_fit"] = {a: spearman(s[torch.from_numpy(found)].numpy(), fit_nll[found])
                                 for a, s in scores.items() if a != "train fit"}
    print("\nSpearman with train fit (seen): "
          + ", ".join(f"{a} {v:+.3f}" for a, v in out["spearman_train_fit"].items()))

    keep = torch.from_numpy(seen)
    y = torch.from_numpy(wrong)[keep]
    feat = {a: (s.clamp_min(1e-12).log() if a not in ("1-conf",) else s)[:, None]
            for a, s in scores.items()}
    soft = torch.stack([strata, torch.tensor(r["probe_scores"]["MAP 1-conf"], dtype=torch.float64)], 1)
    sets = {"softmax": [], "+train fit": ["train fit"]}
    for a in scores:
        if a not in ("train fit", "1-conf"):
            sets[f"+{a}"] = [a]
            sets[f"+train fit+{a}"] = ["train fit", a]
    out["logistic"] = {}
    print("\nheld-out logistic AUROC on wrong (softmax = entropy + 1-conf)")
    for name, ks in sets.items():
        X = torch.cat([soft, *(feat[k] for k in ks)], 1)[keep]
        out["logistic"][name] = heldout_auroc(X, y, pid[seen], args.n_splits)
        print(f"  {name:28s} {out['logistic'][name]:.3f}")

    a1, a2 = list(scores)[:2]
    flag = {a: (s >= torch.quantile(s[keep], 1 - args.top)).numpy() & seen for a, s in scores.items()}
    out["overlap"] = {}
    print(f"\nerrors among the {args.top:.0%} most uncertain seen probes (of {wrong.sum()} errors)")
    for b in [a2, *[a for a in scores if a not in (a1, a2)]]:
        o = {"both": int((wrong & flag[a1] & flag[b]).sum()), f"{a1} only": int((wrong & flag[a1] & ~flag[b]).sum()),
             f"{b} only": int((wrong & ~flag[a1] & flag[b]).sum()), "neither": int((wrong & ~flag[a1] & ~flag[b]).sum())}
        out["overlap"][f"{a1} vs {b}"] = o
        print(f"  {a1} vs {b}: " + ", ".join(f"{k} {v}" for k, v in o.items()))

    out["examples"] = [{"prompt": probes[i]["prompt"], "answer": probes[i]["answer"], "pred": pred_str[i],
                        "freq": int(freq[i]), "type": "swap" if swap[i] else "other",
                        "train_fit": float(fit_nll[i]), **{a: float(s[i]) for a, s in scores.items()}}
                       for i in np.flatnonzero(wrong)[:200]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
