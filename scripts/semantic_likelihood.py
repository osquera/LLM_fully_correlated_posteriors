"""Semantic uncertainty from likelihoods: the epistemic part without sampling
noise.

semantic_uncertainty.py lets every posterior sample draw its own 10 answers,
so two samples differ by chance even when they are the same model, and the
small real disagreement drowns in that noise. Here every model is scored on
the *same* answers instead: the MAP's S*N sampled answers from that run, with
their meaning clusters. For posterior sample s, log p_s(answer | prompt) by
teacher forcing (at the sampling temperature), and importance weights
p_s / p_map turn the MAP's answer counts into s's distribution over meanings
(llmproj.semantic.reweighted_cluster_probs). Then, per question and model:

  sem   total / aleatoric / epistemic over meanings ("semantic MI")
  str   the same with every distinct answer string its own meaning
  lp_std  mean over the answers of the std across samples of log p_s(answer):
          disagreement in nats, no clustering at all

For the MAP every p_s is the same, so its epistemic parts are exactly zero.

  uv run python scripts/semantic_likelihood.py --results figures/semantic_uncertainty.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.paramspace import ParamSpace
from llmproj.semantic import prob_decomposition, reweighted_cluster_probs
from semantic_tables import TASKS, auroc_ci, table, task_records
from semantic_uncertainty import make_prompt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="figures/semantic_uncertainty.json")
    ap.add_argument("--samples", nargs="+",
                    default=["samples/posterior_sequence.pt", "samples/posterior_token.pt"])
    ap.add_argument("--methods", nargs="+", default=None,
                    help="model keys as in --results (default: all of them but the MAP)")
    ap.add_argument("--n-posterior", type=int, default=None, help="default: as in --results")
    ap.add_argument("--batch-answers", type=int, default=32)
    ap.add_argument("--base-model", default=None, help="default: as in --results")
    ap.add_argument("--adapter", default=None, help="default: as in --results")
    ap.add_argument("--out", default=None, help="default: <results>_likelihood.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def answer_logprobs(model, tokenizer, prompt: str, answers: list[str], temperature: float,
                    device: str, batch: int) -> np.ndarray:
    """log p(answer + newline | prompt) at ``temperature``, one per answer.

    Sequences share the prompt and are right-padded, so every answer token
    sits within the last (longest answer + 1) positions: only those logits
    are computed (``logits_to_keep``).
    """
    p_ids = tokenizer(prompt).input_ids
    a_ids = [tokenizer(a + "\n", add_special_tokens=False).input_ids for a in answers]
    out = np.empty(len(answers))
    for i in range(0, len(answers), batch):
        chunk = a_ids[i:i + batch]
        L = max(map(len, chunk))
        ids = torch.full((len(chunk), len(p_ids) + L), tokenizer.pad_token_id)
        mask = torch.zeros_like(ids)
        for k, a in enumerate(chunk):
            ids[k, :len(p_ids) + len(a)] = torch.tensor(p_ids + a)
            mask[k, :len(p_ids) + len(a)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                       logits_to_keep=L + 1).logits[:, :L].float()
        logp = (logits / temperature).log_softmax(-1).cpu()
        for k, a in enumerate(chunk):
            out[i + k] = float(logp[k, torch.arange(len(a)), torch.tensor(a)].sum())
    return out


def main():
    args = parse_args()
    t0 = time.time()
    res = json.loads(Path(args.results).read_text())
    ra, recs = res["args"], res["records"]
    methods = args.methods or [m for m in res["models"] if m != "map"]
    S = args.n_posterior or ra["n_posterior"]
    T = ra["temperature"]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or ra["adapter"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model or ra["base_model"], dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter or ra["adapter"]).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    params = dict(model.named_parameters())
    theta0 = {n: params[n].detach().clone() for n in space.names}

    def set_delta(delta):
        upd = space.unflatten(delta.to(device=space.device, dtype=space.dtype))
        with torch.no_grad():
            for n in space.names:
                params[n].copy_(theta0[n] + upd[n])

    deltas = {}
    for p in args.samples:
        if not Path(p).exists():
            print(f"skipping missing {p}")
            continue
        blob = torch.load(p, weights_only=False)
        mode = blob["info"]["loss_mode"]
        for key, d in blob["methods"].items():
            if f"{mode}/{key}" in methods:
                deltas[f"{mode}/{key}"] = d[:S]
    missing = [m for m in methods if m not in deltas]
    if missing:
        raise SystemExit(f"no samples for {missing}")

    # the shared answer set per question: the MAP's draws, deduplicated
    qs = []
    for r in recs:
        ans = [a for row in r["models"]["map"]["answers"] for a in row]
        cl = [c for row in r["models"]["map"]["clusters"] for c in row]
        cnt = Counter(ans)
        uniq = list(cnt)
        first = {a: cl[ans.index(a)] for a in uniq}
        _, clusters = np.unique([first[a] for a in uniq], return_inverse=True)
        qs.append({"prompt": make_prompt(r["question"], ra.get("few_shot", 0)), "uniq": uniq,
                   "counts": np.array([cnt[a] for a in uniq]), "clusters": clusters})
    print(f"{len(qs)} questions, {np.mean([len(q['uniq']) for q in qs]):.1f} distinct answers each; "
          f"models: map + {methods}")

    def score_all():
        return [answer_logprobs(model, tokenizer, q["prompt"], q["uniq"], T, args.device,
                                args.batch_answers) for q in qs]

    set_delta(torch.zeros(space.P))
    lp_map = score_all()
    logp = {}  # model -> [question] -> (S, U)
    for m in methods:
        rows = []
        for d in deltas[m]:
            set_delta(d)
            rows.append(score_all())
        logp[m] = [np.stack([rows[s][i] for s in range(len(rows))]) for i in range(len(qs))]
        print(f"scored {m}: {len(rows)} samples ({time.time() - t0:.0f} s)")
    set_delta(torch.zeros(space.P))

    for i, (r, q) in enumerate(zip(recs, qs)):
        r["likelihood"] = {}
        for m in methods:
            lp = logp[m][i]
            w = q["counts"] / q["counts"].sum()
            r["likelihood"][m] = {
                "sem": prob_decomposition(reweighted_cluster_probs(lp, lp_map[i], q["counts"], q["clusters"])),
                "str": prob_decomposition(reweighted_cluster_probs(
                    lp, lp_map[i], q["counts"], np.arange(len(q["uniq"])))),
                "lp_std": float((lp.std(0) * w).sum()),
            }
        r["likelihood"]["map_logp"] = lp_map[i].tolist()
        for m in methods:
            r["likelihood"][m]["logp"] = np.round(logp[m][i], 4).tolist()

    out = Path(args.out or str(Path(args.results).with_suffix("")) + "_likelihood.json")
    for r in recs:  # the sampled answers are in --results already
        r.pop("models", None)
    out.write_text(json.dumps({"args": vars(args), "source": args.results, "methods": methods,
                               "records": recs}))
    print(f"wrote {out} ({time.time() - t0:.0f} s)")

    # ---- tables
    rng = np.random.default_rng(0)
    parts = [("sem", "aleatoric", "semantic aleatoric"), ("sem", "epistemic", "semantic MI"),
             ("str", "epistemic", "string MI"), ("lp_std", None, "std of log p")]
    tasks = [t for t in TASKS if len(task_records(recs, t)[0])]
    rows = []
    for m in methods:
        for key, part, name in parts:
            get = (lambda r, k=key, p=part: r["likelihood"][m][k][p]) if part else \
                  (lambda r, k=key: r["likelihood"][m][k])
            cells = []
            for t in tasks:
                rs, lab = task_records(recs, t)
                v, lo, hi = auroc_ci(np.array([get(r) for r in rs]), lab, args.n_boot, rng)
                cells.append("n/a" if np.isnan(v) else f"{v:.3f} [{lo:.2f}, {hi:.2f}]")
            rows.append([f"{m}: {name}"] + cells)
    means = []
    for m in methods:
        for ds in dict.fromkeys(r["dataset"] for r in recs):
            rs = [r for r in recs if r["dataset"] == ds]
            means.append([m, ds] + [f"{np.mean([r['likelihood'][m]['sem'][p] for r in rs]):.2e}"
                                    for p in ("total", "aleatoric", "epistemic")]
                         + [f"{np.mean([r['likelihood'][m]['str']['epistemic'] for r in rs]):.2e}",
                            f"{np.mean([r['likelihood'][m]['lp_std'] for r in rs]):.3g}"])
    md = ["# Likelihood-based semantic uncertainty",
          f"Every model scored on the MAP's {ra['n_posterior']} x {ra['n_answers']} answers per question "
          f"(T = {T}); epistemic parts are exactly 0 for the MAP.",
          "## Detection AUROC [95% bootstrap CI]",
          table(["score"] + [f"{t} ({TASKS[t][2]})" for t in tasks], rows),
          "## Means per dataset",
          table(["model", "dataset", "sem total", "sem aleatoric", "sem epistemic",
                 "string epistemic", "std log p"], means)]
    text = "\n\n".join(md) + "\n"
    print(text)
    out.with_suffix(".md").write_text(text)
    print(f"wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
