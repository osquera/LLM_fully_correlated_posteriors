"""Paired bootstrap of within-entropy AUROC gaps over evaluate_bio.py outputs.

Each output is one replication (its own corpus, theta_map, posterior and
ensemble). Per task, the gap between --ref and each other method is
recomputed on probes resampled with replacement inside every replication,
then averaged over replications: a 95% interval for the mean gap that counts
both probe noise and the spread between replications.

  uv run python scripts/bootstrap_bio.py figures/bio_err_seed{1,2,3}.json \\
      --ref "token/projected_alternating25_measured MI" \\
      --vs "baseline/lora_ensemble MI" "MAP 1-conf" "token/isotropic_measured MI"
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from analyze_posterior import stratified_auroc
from evaluate_bio import FREQ_TASKS


def task_masks(freq, correct):
    tasks = {k: (pos(freq), neg(freq)) for k, (pos, neg) in FREQ_TASKS.items()}
    seen = freq >= 1
    tasks["wrong"] = (seen & ~correct, seen & correct)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--vs", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["unseen", "rare", "wrong"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reps = []
    for path in args.evals:
        r = json.load(open(path))["report"]
        freq, correct = np.array(r["probe_freq"]), np.array(r["probe_correct"], bool)
        reps.append({"path": path, "scores": {k: torch.tensor(v) for k, v in r["probe_scores"].items()},
                     "tasks": task_masks(freq, correct)})
    methods = [args.ref, *args.vs]
    rng = np.random.default_rng(args.seed)
    out = {"evals": args.evals, "ref": args.ref, "tasks": {}}
    for t in args.tasks:
        per_rep, boots = [], []
        for rep in reps:
            pos, neg = rep["tasks"][t]
            keep = torch.from_numpy(pos | neg)
            y = torch.from_numpy(pos)[keep]
            x = {m: rep["scores"][m][keep] for m in methods}
            strata = rep["scores"]["MAP entropy"][keep]

            def aurocs(i):
                return {m: stratified_auroc(y[i], x[m][i], strata[i], args.n_bins)["weighted"]
                        for m in methods}

            per_rep.append({"n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                            **aurocs(torch.arange(len(y)))})
            b = []
            for _ in range(args.n_boot):
                a = aurocs(torch.from_numpy(rng.integers(0, len(y), len(y))))
                b.append([a[args.ref] - a[m] for m in args.vs])
            boots.append(np.array(b))
        gaps = np.mean(boots, axis=0)  # (n_boot, n_vs): mean gap over replications
        point = {m: float(np.mean([p[args.ref] - p[m] for p in per_rep])) for m in args.vs}
        res = {"per_replication": per_rep, "gaps": {}}
        print(f"\n{t}: positives {[p['n_pos'] for p in per_rep]}, within-entropy AUROC per replication")
        for m in methods:
            print(f"  {m:45s} " + "  ".join(f"{p[m]:.3f}" for p in per_rep))
        print(f"  {args.ref} minus:")
        for j, m in enumerate(args.vs):
            lo, hi = np.quantile(gaps[:, j], [0.025, 0.975])
            p0 = float(np.mean(gaps[:, j] <= 0))
            res["gaps"][m] = {"mean": point[m], "ci95": [float(lo), float(hi)], "p_le_0": p0}
            print(f"    {m:43s} {point[m]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(<=0) {p0:.3f}")
        out["tasks"][t] = res
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
