"""Copy scores from one evaluate_bio.py output into another on the same probes.

For methods scored in a separate run (e.g. a bagged ensemble, E49): every
score whose name starts with one of --prefix is copied from --other into
--base (task AUROCs, logistic gains, probe scores and subset summaries), so
bootstrap_bio.py can compare it with the base file's methods probe for probe.

  uv run python scripts/merge_eval_scores.py --base figures/e49_tofuh1.json \\
      --other figures/e49_tofuh1_bagged.json --prefix baseline/bagged_ensemble \\
      --out figures/e49_tofuh1.json
"""

from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--prefix", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base, other = json.load(open(args.base)), json.load(open(args.other))
    b, o = base["report"], other["report"]
    if b["probe_freq"] != o["probe_freq"] or b.get("probe_name") != o.get("probe_name"):
        raise SystemExit("the two outputs are not on the same probes")
    hit = lambda k: any(k.startswith(p) or k.startswith(f"softmax+{p}") for p in args.prefix)  # noqa: E731
    n = 0
    for t, d in o["tasks"].items():
        for k, v in d.items():
            if hit(k):
                b["tasks"].setdefault(t, {})[k] = v
                n += 1
    for t, d in o["logistic"].items():
        b["logistic"].setdefault(t, {}).update({k: v for k, v in d.items() if hit(k)})
    b["probe_scores"].update({k: v for k, v in o["probe_scores"].items() if hit(k)})
    b.setdefault("subsets", {}).update({k: v for k, v in o.get("subsets", {}).items() if hit(k)})
    json.dump(base, open(args.out, "w"))
    print(f"merged {n} task scores for {args.prefix} into {args.out}")


if __name__ == "__main__":
    main()
