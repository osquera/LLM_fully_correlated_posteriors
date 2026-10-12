"""Plot figures/tolerance_sweep.json (from sweep_tolerance.py), one line per method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PANELS = [  # (title, getter, log y)
    ("train: worst |Δ seq NLL| (Lemma 4.3)", lambda r: r["train"]["worst_seq_dnll"], True),
    ("test: BMA NLL", lambda r: r["test"]["nll_bma"], False),
    ("test: ECE", lambda r: r["test"]["ece"], False),
    ("OOD detection AUROC (MI)", lambda r: r["ood"]["auroc"], False),
    ("test+ood: error AUROC by MI", lambda r: r["pooled"]["selective"].get("mi", {}).get("err_auroc"), False),
    ("test+ood: AURC by entropy (lower better)", lambda r: r["pooled"]["selective"]["entropy"]["aurc"], False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="figures/tolerance_sweep.json")
    ap.add_argument("--out", default="figures/tolerance_sweep.png")
    args = ap.parse_args()
    sweep = json.loads(Path(args.sweep).read_text())["sweep"]
    tols = [s["tol"] for s in sweep]
    methods = list(sweep[0]["results"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, get, logy) in zip(axes.flat, PANELS):
        for m in methods:
            ys = [get(s["results"][m]) for s in sweep]
            if all(y is None for y in ys):
                continue
            style = "--" if m == "map" else "-o" if "projected" in m else "-."
            ax.plot(tols, ys, style, label=m, ms=4, color=f"C{methods.index(m)}")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("tolerance (fraction of MAP mean NLL)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
