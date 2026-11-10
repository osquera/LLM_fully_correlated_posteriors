"""Facts or wording? The paraphrase test (E51).

On evaluate_bio.py outputs for make_phrasing_corpus.py corpora (every fact in
every phrasing, with membership scores merged in), per score:

  untrained vs trained, seen wording     the usual unseen task
  untrained vs trained, unseen wording   trained facts asked in words never
                                         trained: a wording tracker loses its
                                         signal here, a fact tracker keeps it
  wording only                           trained facts, unseen vs seen
                                         wording: ~0.5 for a pure fact
                                         tracker, high for a wording tracker
  wrong, unseen wording                  errors among trained facts asked in
                                         unseen wording

All within MAP-entropy quintiles. Trained probes are restricted to
--max-freq (default 2: the frequencies where both wordings occur), so
frequency does not differ between the seen- and unseen-wording groups.
A person-cluster bootstrap gives the gap --ref minus each other score on the
unseen-wording task, pooled over replications.

  uv run python scripts/analyze_phrasings.py --evals figures/e51_bio{1,2,3}.json \\
      --corpora data/bio_err{1,2,3}/phrasings.json --out figures/e51_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_posterior import stratified_auroc

DEFAULT_SCORES = ["token/projected_measured MI", "baseline/lora_ensemble MI",
                  "token/isotropic_measured MI", "MAP 1-conf", "question NLL",
                  "question min-20%", "question NLL ratio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--scores", nargs="+", default=DEFAULT_SCORES)
    ap.add_argument("--ref", default="token/projected_measured MI")
    ap.add_argument("--max-freq", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reps = []
    for ev, cp in zip(args.evals, args.corpora):
        r = json.load(open(ev))["report"]
        probes = json.loads(Path(cp).read_text())["probes"]
        if [q["freq"] for q in probes] != r["probe_freq"]:
            raise SystemExit(f"{ev} and {cp} differ")
        freq = np.array(r["probe_freq"])
        seen = np.array([q["seen"] for q in probes])
        correct = np.array(r["probe_correct"], bool)
        trained = (freq >= 1) & (freq <= args.max_freq)
        untrained = freq == 0
        tasks = {
            "untrained vs trained, seen wording": (untrained, trained & seen),
            "untrained vs trained, unseen wording": (untrained, trained & ~seen),
            "wording only (unseen vs seen, trained)": (trained & ~seen, trained & seen),
            "wrong, unseen wording": (trained & ~seen & ~correct, trained & ~seen & correct),
        }
        _, pid = np.unique(np.array(r["probe_name"]), return_inverse=True)
        S = {k: torch.tensor(v, dtype=torch.float64) for k, v in r["probe_scores"].items()}
        missing = [k for k in args.scores if k not in S]
        if missing:
            raise SystemExit(f"{ev}: no scores {missing}")
        reps.append({"tasks": tasks, "S": S, "pid": pid, "strata": S["MAP entropy"]})

    rng = np.random.default_rng(args.seed)
    out = {"evals": args.evals, "max_freq": args.max_freq, "tasks": {}}
    for t in reps[0]["tasks"]:
        per = {k: [] for k in args.scores}
        boots = []
        for rep in reps:
            pos, neg = rep["tasks"][t]
            keep = pos | neg
            k = torch.from_numpy(keep)
            y = torch.from_numpy(pos[keep])
            st = rep["strata"][k]
            xs = {s: rep["S"][s][k] for s in args.scores}

            def au(idx):
                return {s: stratified_auroc(y[idx], xs[s][idx], st[idx], args.n_bins)["weighted"]
                        for s in args.scores}

            a = au(torch.arange(len(y)))
            for s in args.scores:
                per[s].append(a[s])
            if t == "untrained vs trained, unseen wording":
                _, g = np.unique(rep["pid"][keep], return_inverse=True)
                groups = [np.flatnonzero(g == i) for i in range(g.max() + 1)]
                b = []
                for _ in range(args.n_boot):
                    idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                    ab = au(torch.from_numpy(idx))
                    b.append([ab[args.ref] - ab[s] for s in args.scores])
                boots.append(np.array(b))
        n = [int(rep["tasks"][t][0].sum()) for rep in reps], [int(rep["tasks"][t][1].sum()) for rep in reps]
        print(f"\n{t}  (positives {n[0]}, negatives {n[1]})")
        res = {"n_pos": n[0], "n_neg": n[1], "auroc": {}}
        for s in args.scores:
            res["auroc"][s] = {"mean": float(np.mean(per[s])), "per_replication": per[s]}
            print(f"  {s:32s} {np.mean(per[s]):.3f}  ({', '.join(f'{x:.3f}' for x in per[s])})")
        if boots:
            gaps = np.mean(boots, axis=0)
            res["gap_vs_ref"] = {}
            print(f"  {args.ref} minus (person-cluster bootstrap):")
            for j, s in enumerate(args.scores):
                if s == args.ref:
                    continue
                lo, hi = np.quantile(gaps[:, j], [0.025, 0.975])
                pt = float(np.mean(per[args.ref]) - np.mean(per[s]))
                res["gap_vs_ref"][s] = {"mean": pt, "ci95": [float(lo), float(hi)]}
                print(f"    {s:30s} {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        out["tasks"][t] = res
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
