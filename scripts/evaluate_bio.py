"""Does uncertainty track how well the training data supports a fact?

On the synthetic biography corpus (scripts/make_bio_corpus.py), every probe
"<name> was born in" -> " <city>" has a known training frequency f (the number
of biographies of that person seen in fine-tuning, 0 = never). Teacher-forced
on the true answer, per probe (mean over its answer tokens):

  MAP     entropy, 1 - confidence, exact-match correctness (every answer token
          the argmax)
  each posterior / baseline samples file method: MI (the samples'
          disagreement)
  hidden-state k-NN and Mahalanobis distance (theta_map, middle and last
          layer), fitted on the corpus tokens

Tasks, AUROC raw and within quintiles of the MAP's entropy (the second asks
whether a score knows something the softmax does not):
  unseen      f = 0 vs f >= 1
  rare        f <= 1 vs f >= 8
  wrong       MAP exact match wrong vs right, among seen people (f >= 1)
plus Spearman rho between the score and -log2(1 + f), and a held-out
logistic regression per task (2 folds split by person): MAP entropy + 1-conf,
with and without each method's log MI (does MI add to the softmax as a whole).

--norms rescales each token-mode posterior method to these sample norms
(``name@norm``), and every method's fit cost is reported: per-document mean
NLL change on the corpus vs theta_map (worst, mean |.|, share over the
sampler's tolerance). Per-probe scores go into the output for later analysis.

  uv run python scripts/evaluate_bio.py --corpus data/bio/corpus.json \\
      --adapter checkpoints/bio_lora --samples samples/posterior_token_bio.pt samples/baselines_bio.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_posterior import logistic_gain, seq_index, stratified_auroc
from baselines_single_pass import KNN, Mahalanobis, seq_mean, token_features
from evaluate_posterior import auroc, evaluate_set
from llmproj.data import encode_texts
from llmproj.paramspace import ParamSpace

FREQ_TASKS = {"unseen": (lambda f: f == 0, lambda f: f >= 1),
              "rare": (lambda f: f <= 1, lambda f: f >= 8)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/bio/corpus.json")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/bio_lora")
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--methods", nargs="*", default=None,
                    help="method keys (default: every *_measured one and every baseline)")
    ap.add_argument("--norms", type=float, nargs="*", default=[],
                    help="also evaluate token-mode posterior methods rescaled to these ||delta||")
    ap.add_argument("--max-length", type=int, default=96, help="corpus encoding for the fit cost")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="figures/bio_eval.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def encode_probes(tokenizer, probes):
    """prompt + answer, loss_mask on the answer tokens, right-padded."""
    rows = []
    for q in probes:
        p = tokenizer(q["prompt"]).input_ids
        a = tokenizer(q["answer"], add_special_tokens=False).input_ids
        rows.append((p + a, [0] * len(p) + [1] * len(a)))
    T = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), T), tokenizer.pad_token_id)
    att, lm = torch.zeros(len(rows), T, dtype=torch.long), torch.zeros(len(rows), T, dtype=torch.long)
    for i, (x, m) in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
        lm[i, :len(x)] = torch.tensor(m)
    return {"input_ids": ids, "attention_mask": att, "loss_mask": lm}


def per_probe(values: torch.Tensor, enc) -> torch.Tensor:
    """Mean of a per-answer-token value over each probe's answer tokens."""
    sid = seq_index(enc)
    n = enc["input_ids"].shape[0]
    tot = torch.zeros(n).index_add_(0, sid, values.float())
    cnt = torch.zeros(n).index_add_(0, sid, torch.ones(len(values)))
    return tot / cnt


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    args = parse_args()
    corpus = json.loads(Path(args.corpus).read_text())
    probes = corpus["probes"]
    freq = np.array([q["freq"] for q in probes])

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, attn_implementation="eager",
                                                dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta, frozen = space.theta(model), space.frozen(model)
    enc = encode_probes(tokenizer, probes)

    methods, tol = {"map": torch.zeros(1, space.P)}, None
    for p in args.samples:
        blob = torch.load(p, weights_only=False)
        mode = blob["info"]["loss_mode"]
        if mode == "token":
            tol = blob["info"]["alpha_tol_nats"]
        for m, d in blob["methods"].items():
            key = f"{mode}/{m}"
            if (args.methods and key not in args.methods) or \
               (not args.methods and not (m.endswith("_measured") or mode == "baseline")):
                continue
            methods[key] = d.float()
            if mode == "token":
                for n in args.norms:
                    methods[f"{key}@{n:g}"] = d.float() * (n / float(d.norm(dim=1).mean()))
    print(f"{len(probes)} probes; methods {list(methods)}")

    corpus_enc = encode_texts(tokenizer, corpus["texts"], args.max_length, loss_on="all")
    scores, map_tok, fit, map_doc = {}, None, {}, None
    for name, d in methods.items():
        _, s, _ = evaluate_set(model, frozen, theta, space, d, enc, args.device, args.batch_size)
        _, _, doc_nll = evaluate_set(model, frozen, theta, space, d, corpus_enc, args.device, args.batch_size)
        if name == "map":
            map_tok, map_doc = s["tok"], doc_nll[0]
            scores["MAP entropy"] = per_probe(s["tok"]["entropy"], enc)
            scores["MAP 1-conf"] = per_probe(1 - s["tok"]["conf"], enc)
        else:
            scores[f"{name} MI"] = per_probe(s["tok"]["mi"], enc)
            dn = doc_nll - map_doc
            fit[name] = {"norm": float(d.norm(dim=1).mean()), "mean_dnll": float(dn.mean()),
                         "worst_dnll": float(dn.abs().max()), "mean_abs_dnll": float(dn.abs().mean()),
                         "frac_over_tol": float((dn.abs() > tol).float().mean()) if tol else None}
        print(f"  scored {name}")
    correct = per_probe(map_tok["correct"].float(), enc).eq(1).numpy()  # exact match

    # hidden-state distances, fitted on the corpus tokens
    n_hidden = model.config.num_hidden_layers
    layers = args.layers or [n_hidden // 2, n_hidden]
    train_enc = encode_texts(tokenizer, corpus["texts"], 128, loss_on="all")
    f_train, _ = token_features(model, train_enc, layers, args.device, args.batch_size)
    f_probe, _ = token_features(model, enc, layers, args.device, args.batch_size)
    for l in layers:
        scores[f"Mahalanobis L{l}"] = seq_mean(Mahalanobis(f_train[l], 1e-3, args.device)(f_probe[l]), enc)
        scores[f"k-NN L{l}"] = seq_mean(KNN(f_train[l], args.knn_k, 50_000, args.device)(f_probe[l]), enc)

    strata = scores["MAP entropy"]
    groups = sorted(set(freq.tolist()))
    report = {"per_freq": {}, "tasks": {}, "spearman": {}, "logistic": {}, "fit": fit,
              "tol_nats": tol, "map_corpus_nll": float(map_doc.mean()), "n_probes": len(probes),
              "layers": layers}
    for f in groups:
        m = freq == f
        report["per_freq"][f] = {"n": int(m.sum()), "map_exact_acc": float(correct[m].mean()),
                                 **{k: float(v[torch.from_numpy(m)].mean()) for k, v in scores.items()}}
    tasks = {k: (pos(freq), neg(freq)) for k, (pos, neg) in FREQ_TASKS.items()}
    seen = freq >= 1
    tasks["wrong"] = (seen & ~correct, seen & correct)
    for t, (pos, neg) in tasks.items():
        keep = torch.from_numpy(pos | neg)
        y = torch.from_numpy(pos)[keep]
        report["tasks"][t] = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
        for k, v in scores.items():
            x = v[keep]
            report["tasks"][t][k] = {
                "raw": auroc(x[~y], x[y]),
                "within_entropy": stratified_auroc(y, x, strata[keep], args.n_bins)["weighted"]}
    for k, v in scores.items():
        report["spearman"][k] = spearman(v.numpy(), -np.log2(1 + freq))
    # does MI add to the softmax as a whole? folds split by person
    person = {n: i for i, n in enumerate(dict.fromkeys(q["name"] for q in probes))}
    pid = torch.tensor([person[q["name"]] for q in probes])
    soft = torch.stack([scores["MAP entropy"], scores["MAP 1-conf"]], 1).double()
    mi_keys = [k for k in scores if k.endswith(" MI")]
    for t, (pos, neg) in tasks.items():
        keep = torch.from_numpy(pos | neg)
        y = torch.from_numpy(pos)[keep]
        feats = {"softmax": soft[keep]}
        for k in mi_keys:
            lmi = scores[k].double().clamp_min(1e-12).log()[keep, None]
            feats[f"softmax+{k}"] = torch.cat([soft[keep], lmi], 1)
        report["logistic"][t] = {k: v["auroc"] for k, v in logistic_gain(y, feats, pid[keep]).items()}
    report["probe_scores"] = {k: v.tolist() for k, v in scores.items()}
    report["probe_freq"], report["probe_correct"] = freq.tolist(), correct.tolist()

    print("\nper training frequency: MAP exact-match accuracy and mean scores")
    keys = list(scores)
    print(f"  {'f':>3s} {'n':>4s} {'acc':>6s} " + " ".join(f"{k[:22]:>22s}" for k in keys))
    for f, r in report["per_freq"].items():
        print(f"  {f:3d} {r['n']:4d} {r['map_exact_acc']:6.3f} " +
              " ".join(f"{r[k]:22.4g}" for k in keys))
    for t, r in report["tasks"].items():
        print(f"\n{t}: {r['n_pos']} positive / {r['n_neg']} negative; AUROC raw / within entropy quintiles")
        for k in keys:
            print(f"  {k:40s} {r[k]['raw']:.3f} / {r[k]['within_entropy']:.3f}")
    print("\nSpearman(score, rarity):")
    for k in keys:
        print(f"  {k:40s} {report['spearman'][k]:+.3f}")
    print("\nheld-out logistic AUROC, softmax (entropy + 1-conf) -> + log MI:")
    for t, r in report["logistic"].items():
        print(f"  {t}: softmax {r['softmax']:.3f}")
        for k, v in r.items():
            if k != "softmax":
                print(f"    {k[8:]:40s} {v:.3f}")
    print(f"\nfit cost on the corpus (MAP mean NLL {report['map_corpus_nll']:.4f}, tol {tol}):")
    print(f"  {'method':40s} {'norm':>7s} {'mean dNLL':>10s} {'worst':>8s} {'over tol':>8s}")
    for k, r in fit.items():
        ot = f"{r['frac_over_tol']:8.3f}" if r["frac_over_tol"] is not None else "       -"
        print(f"  {k:40s} {r['norm']:7.3g} {r['mean_dnll']:10.4f} {r['worst_dnll']:8.4f} {ot}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "args": vars(args)}, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
