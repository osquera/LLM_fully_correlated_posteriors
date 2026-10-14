"""Mean ± std over seeds of evaluate_posterior.py outputs (one JSON per seed).

  uv run python scripts/aggregate_seeds.py figures/posterior_eval_seed*.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

METRICS = {  # column -> getter
    "norm": lambda r: r["norm"],
    "train worst Δ": lambda r: r["train"]["worst_seq_dnll"],
    "train mean |Δ|": lambda r: r["train"]["mean_abs_seq_dnll"],
    "test NLL": lambda r: r["test"]["nll_bma"],
    "test ECE": lambda r: r["test"]["ece"],
    "test MI": lambda r: r["test"]["mi"],
    "ood MI": lambda r: r["ood"]["mi"],
    "OOD AUROC": lambda r: r["ood"]["auroc"],
    "err AUROC (MI)": lambda r: r["pooled"]["selective"].get("mi", {}).get("err_auroc"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [json.loads(Path(p).read_text())["results"] for p in args.evals]
    methods = [m for m in runs[0] if all(m in r for r in runs)]

    summary = {}
    for m in methods:
        summary[m] = {}
        for col, get in METRICS.items():
            vals = [get(r[m]) for r in runs]
            vals = [v for v in vals if v is not None and not math.isnan(v)]  # nan: one class
            if not vals:
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary[m][col] = {"mean": statistics.fmean(vals), "std": sd, "values": vals}

    print(f"{len(runs)} seeds; mean ± std")
    cols = list(METRICS)
    print(f"{'method':36s} " + " ".join(f"{c:>20s}" for c in cols))
    for m, s in summary.items():
        cells = [f"{s[c]['mean']:.4g} ± {s[c]['std']:.2g}" if c in s else "-" for c in cols]
        print(f"{m:36s} " + " ".join(f"{c:>20s}" for c in cells))
    if args.out:
        Path(args.out).write_text(json.dumps({"files": args.evals, "summary": summary}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
