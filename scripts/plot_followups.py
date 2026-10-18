"""Plot the follow-up analyses: figures/analysis.json (analyze_posterior.py)
and figures/posterior_eval_seeds.json (aggregate_seeds.py).

  a  is the model wrong where uncertain? error AUROC (target: MAP wrong) from
     MI vs the MAP's own 1 - confidence, test + near-OOD tokens
  b  does MI add anything at fixed entropy? AUROC of MI within each entropy
     decile (0.5 = no information; below = high MI marks *correct* tokens)
  c  Lemma 4.3 off the projection set: fraction of train sequences whose NLL
     moves more than the tolerance, constrained (256) vs never projected (1744)
  d  near-OOD detection: AUROC vs the Alpaca test set, by MI (MAP: entropy)
  e, f  5 seeds (sequence mode): mean ± std and the individual seeds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLOR = {"projected": "C2", "isotropic": "C7", "diagonal_laplace": "C1", "lla": "C4"}
SETS = ["wikitext", "dolly", "codealpaca", "gsm8k", "alpaca_es"]


def label(m):
    if m == "map":
        return "MAP"
    mode, base = m.split("/")
    return f"{mode} / {base.removesuffix('_measured').replace('_', ' ')}"


def color(m):
    return "k" if m == "map" else COLOR[m.split("/")[1].removesuffix("_measured")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="figures/analysis.json")
    ap.add_argument("--seeds", default="figures/posterior_eval_seeds.json")
    ap.add_argument("--pool", default="test+near_ood", choices=["test", "test+wikitext", "test+near_ood"])
    ap.add_argument("--out", default="figures/followups.png")
    args = ap.parse_args()
    rep = json.loads(Path(args.analysis).read_text())["report"]
    methods = [m for m in rep["methods"] if m != "map"]
    mib, unc, near = rep["mi_beyond_softmax"], rep["unconstrained_train"], rep["near_ood"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    (a, b, c), (d, e, f) = axes
    y = np.arange(len(methods))

    # a: raw error AUROC
    mi = [mib[m][args.pool]["raw_auroc"]["mi"] for m in methods]
    a.barh(y, mi, color=[color(m) for m in methods])
    ref = mib[methods[0]][args.pool]["raw_auroc"]
    a.axvline(ref["map_1-conf"], c="k", ls="--", lw=1.2, label=f"MAP 1-conf {ref['map_1-conf']:.3f}")
    a.axvline(ref["map_entropy"], c="k", ls=":", lw=1.2, label=f"MAP entropy {ref['map_entropy']:.3f}")
    a.axvline(0.5, c="0.6", lw=0.8)
    for i, v in enumerate(mi):
        a.text(v - 0.01, i, f"{v:.3f}", va="center", ha="right", color="w", fontsize=9)
    a.set(yticks=y, yticklabels=[label(m) for m in methods], xlim=(0.45, 1), xlabel="error AUROC by MI")
    a.invert_yaxis()
    a.legend(loc="lower right", fontsize=8)
    a.set_title(f"a  is MI high where the MAP is wrong? ({args.pool})", loc="left")

    # b: MI within entropy deciles
    for m in methods:
        s = mib[m][args.pool]["mi_within_entropy_bins"]
        proj = "projected" in m
        b.plot(range(1, 11), s["per_bin"], color=color(m), ls="-" if m.startswith("sequence") else "--",
               marker="o" if proj else ".", lw=2.2 if proj else 1.2,
               label=f"{label(m)} (weighted {s['weighted']:.2f})")
    err = mib[methods[0]][args.pool]["mi_within_entropy_bins"]["bin_error_rate"]
    b.axhline(0.5, c="k", ls=":", lw=1)
    bt = b.twinx()
    bt.bar(range(1, 11), err, color="0.85", zorder=0, width=0.6)
    bt.set(ylim=(0, 3), yticks=[0, 0.5, 1], ylabel="MAP error rate in decile (grey bars)")
    b.set_zorder(bt.get_zorder() + 1)
    b.patch.set_visible(False)
    b.set(xlabel="BMA entropy decile (1 = most certain)", ylabel="AUROC of MI within decile",
          xticks=range(1, 11), ylim=(0, 1.0))
    b.legend(fontsize=7.5, loc="upper right")
    b.set_title("b  MI at fixed entropy: < 0.5 = high MI on correct tokens", loc="left")

    # c: constrained vs never-projected train sequences
    over = lambda m, k: 100 * unc[m][k]["frac_seq_over_tol"]
    w = 0.38
    c.barh(y - w / 2, [over(m, "train_constrained") for m in methods], w,
           color=[color(m) for m in methods], label="constrained (256)")
    c.barh(y + w / 2, [over(m, "train_unconstrained") for m in methods], w,
           color=[color(m) for m in methods], alpha=0.45, hatch="//", label="never projected (1744)")
    for i, m in enumerate(methods):
        for dy, k in ((-w / 2, "train_constrained"), (w / 2, "train_unconstrained")):
            c.text(over(m, k) + 1, i + dy, f"{over(m, k):.1f}%  (max {unc[m][k]['max']:.3f})",
                   va="center", fontsize=8)
    c.set(yticks=y, yticklabels=[label(m) for m in methods], xlim=(0, 150),
          xlabel=f"% train sequences with |Δ NLL| > tol ({rep['tol_nats']:.4f} nats)")
    c.invert_yaxis()
    c.legend(loc="upper right", fontsize=8)
    c.set_title("c  Lemma 4.3 off the projection set", loc="left")

    # d: near-OOD AUROC heat map
    rows = ["map"] + methods
    grid = np.array([[near[m][s]["auroc_entropy" if m == "map" else "auroc_mi"] for s in SETS]
                     for m in rows])
    im = d.imshow(grid, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(SETS)):
            v = grid[i, j]
            d.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                   color="w" if abs(v - 0.5) > 0.3 else "k")
    d.set(xticks=range(len(SETS)), xticklabels=SETS, yticks=range(len(rows)),
          yticklabels=["MAP (entropy)"] + [label(m) for m in methods])
    d.tick_params(axis="x", labelrotation=30)
    fig.colorbar(im, ax=d, fraction=0.04, label="AUROC vs Alpaca test (0.5 = chance)")
    d.set_title("d  OOD / near-OOD detection by MI", loc="left")

    # e, f: seeds
    if Path(args.seeds).exists():
        summ = json.loads(Path(args.seeds).read_text())["summary"]
        sm = [m for m in summ if m != "map" and "closed_form" not in m]
        ys = np.arange(len(sm))
        for ax, col, title, ref in (
                (e, "err AUROC (MI)", "e  error AUROC by MI, pooled test + WikiText", None),
                (f, "OOD AUROC", "f  WikiText OOD AUROC by MI", summ["map"]["OOD AUROC"]["mean"])):
            for i, m in enumerate(sm):
                s = summ[m][col]
                ax.errorbar(s["mean"], i, xerr=s["std"], fmt="o", color=color(m), ms=8, capsize=4)
                ax.scatter(s["values"], [i + 0.18] * len(s["values"]), color=color(m), s=12, alpha=0.6)
                ax.text(s["mean"], i - 0.25, f"{s['mean']:.3f} ± {s['std']:.3f}", ha="center", fontsize=8)
            if ref is not None:
                ax.axvline(ref, c="k", ls="--", lw=1, label=f"MAP entropy {ref:.3f}")
                ax.legend(loc="center right", fontsize=8)
            ax.axvline(0.5, c="0.6", lw=0.8)
            n = len(summ[sm[0]][col]["values"])
            ax.set(yticks=ys, yticklabels=[label(m) for m in sm], xlabel=f"{col}, {n} seeds",
                   ylim=(len(sm) - 0.5, -0.6))
            ax.set_title(title, loc="left")
        e.set_xlim(0.6, 0.75)
    else:
        for ax in (e, f):
            ax.set_axis_off()

    fig.suptitle(f"Follow-up analyses (loss on {rep['loss_on']}, measured α, tol 1%)", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
