"""Figures for one run of evaluate_posterior.py (figures/posterior_eval.json).

Six panels:
  a  Lemma 4.3: worst and mean change in a train sequence's NLL, per method at
     the measured alpha, against the tolerance alpha was measured at
  b  sample size vs held-out accuracy, every method: the closed-form trap
  c  held-out NLL and ECE of the measured-alpha methods, relative to the MAP
  d  epistemic uncertainty (MI) on test vs OOD, and the OOD AUROC it gives
  e  the measured radii per direction, per loss mode
  f  cost of the projection, per loss mode

Usage:
  uv run python scripts/plot_results.py
  uv run python scripts/plot_results.py --eval hpc_results/posterior_eval.json \\
      --sampler hpc_results/posterior_sequence.json hpc_results/posterior_token.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODE_COLOR = {"sequence": "C0", "token": "C3"}
METHOD_COLOR = {
    "projected": "C2", "isotropic": "C7", "diagonal_laplace": "C1", "lla": "C4",
}


def split(name: str) -> tuple[str, str, str]:
    """'sequence/projected_measured' -> ('sequence', 'projected', 'measured')."""
    if name == "map":
        return "map", "map", "-"
    mode, m = name.split("/")
    for alpha in ("measured", "closed_form"):
        if m.endswith("_" + alpha):
            return mode, m[: -len(alpha) - 1], alpha
    return mode, m, "-"


def label(name: str) -> str:
    mode, base, alpha = split(name)
    return "MAP" if mode == "map" else f"{mode[:3]} / {base.replace('_', ' ')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="figures/posterior_eval.json")
    ap.add_argument("--sampler", nargs="+",
                    default=["samples/posterior_sequence.json", "samples/posterior_token.json"])
    ap.add_argument("--out", default="figures/first_run.png")
    args = ap.parse_args()

    res = json.loads(Path(args.eval).read_text())["results"]
    samp = {}
    for p in args.sampler:
        if Path(p).exists():
            info = json.loads(Path(p).read_text())["info"]
            samp[info["loss_mode"]] = info
    measured = [n for n in res if split(n)[2] == "measured"]
    map_r = res["map"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    (a, b, c), (d, e, f) = axes

    # a: Lemma 4.3 at matched norm
    y = np.arange(len(measured))
    worst = [res[n]["train"]["worst_seq_dnll"] for n in measured]
    mean = [res[n]["train"]["mean_abs_seq_dnll"] for n in measured]
    cols = [METHOD_COLOR[split(n)[1]] for n in measured]
    a.barh(y + 0.2, worst, 0.4, color=cols, label="worst sequence")
    a.barh(y - 0.2, mean, 0.4, color=cols, alpha=0.4, label="mean |Δ|")
    for yi, n, w in zip(y, measured, worst):
        a.text(w * 1.08, yi + 0.2, f"{w:.3f}  (‖δ‖ {res[n]['norm']:.2f})", va="center", fontsize=8)
    tol = next(iter(samp.values()), {}).get("alpha_tol_nats")
    if tol:
        a.axvline(tol, ls="--", c="k", lw=1)
        a.text(tol, -0.6, f" tol {tol:.4f}", fontsize=8)
    a.set_xscale("log")
    a.set_yticks(y, [label(n) for n in measured], fontsize=9)
    a.invert_yaxis()
    a.set_xlim(right=max(worst) * 20)
    a.set_xlabel("|change in a train sequence's mean NLL| (nats)")
    a.set_title("a  Lemma 4.3: train loss under samples, measured α", loc="left")
    a.legend(fontsize=8, loc="upper right")

    # b: closed-form trap
    for n, r in res.items():
        mode, base, alpha = split(n)
        if mode == "map":
            continue
        b.scatter(r["norm"], r["test"]["acc"], s=70,
                  marker="o" if alpha == "measured" else "X",
                  color=METHOD_COLOR[base], edgecolor=MODE_COLOR[mode], linewidth=1.5)
    b.axhline(map_r["test"]["acc"], ls="--", c="k", lw=1, label="MAP")
    for i, (mode, info) in enumerate(samp.items()):
        for kind, a_ in info["alphas"].items():
            norm = (info["kernel_dim"] / a_) ** 0.5
            acc = res[f"{mode}/projected_{kind}"]["test"]["acc"]
            yy = acc - 0.08 - 0.07 * i if kind == "measured" else acc + 0.04
            b.annotate(f"{mode} α={a_:.3g}", (norm, yy), fontsize=7, ha="center",
                       color=MODE_COLOR[mode])
    b.set_xscale("log")
    b.set_xlabel("mean sample norm ‖δ‖")
    b.set_ylabel("held-out next-token accuracy (BMA)")
    b.set_title("b  closed-form α* (×) vs measured α (●)", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c_, label=m.replace("_", " "))
               for m, c_ in METHOD_COLOR.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="w", mec=c_, mew=1.5, label=f"{m} mode")
                for m, c_ in MODE_COLOR.items()]
    b.legend(handles=handles + [plt.Line2D([], [], ls="--", c="k", label="MAP")], fontsize=7,
             loc="center left")

    # c: held-out quality relative to MAP
    dn = [1e3 * (res[n]["test"]["nll_bma"] - map_r["test"]["nll_bma"]) for n in measured]
    de = [1e3 * (res[n]["test"]["ece"] - map_r["test"]["ece"]) for n in measured]
    c.barh(y + 0.2, dn, 0.4, color=cols, label="Δ test NLL")
    c.barh(y - 0.2, de, 0.4, color=cols, alpha=0.4, hatch="//", label="Δ test ECE")
    c.axvline(0, c="k", lw=1)
    c.set_yticks(y, [label(n) for n in measured], fontsize=9)
    c.invert_yaxis()
    c.set_xlabel("change vs MAP (×10⁻³; lower is better)")
    c.set_title(f"c  held-out: MAP NLL {map_r['test']['nll_bma']:.4f}, "
                f"ECE {map_r['test']['ece']:.4f}", loc="left")
    c.legend(fontsize=8)

    # d: MI test vs ood
    d.barh(y + 0.2, [res[n]["test"]["mi"] for n in measured], 0.4, color="C0", label="test (Alpaca)")
    d.barh(y - 0.2, [res[n]["ood"]["mi"] for n in measured], 0.4, color="C3", label="OOD (WikiText)")
    for yi, n in zip(y, measured):
        hi = max(res[n]["test"]["mi"], res[n]["ood"]["mi"])
        d.text(hi * 1.05, yi, f"AUROC {res[n]['ood']['auroc']:.2f}", va="center", fontsize=8)
    d.set_yticks(y, [label(n) for n in measured], fontsize=9)
    d.invert_yaxis()
    d.set_xlim(right=max(max(res[n]["test"]["mi"], res[n]["ood"]["mi"]) for n in measured) * 1.35)
    d.set_xlabel("mean mutual information per token (nats)")
    d.set_title(f"d  epistemic uncertainty (MAP entropy AUROC "
                f"{map_r['ood']['auroc']:.3f})", loc="left")
    d.legend(fontsize=8, loc="center right")

    # e: radii
    for i, (mode, info) in enumerate(samp.items()):
        r = np.array(info["radii"])
        e.scatter(np.full(len(r), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(r)), r,
                  color=MODE_COLOR[mode], alpha=0.7)
        e.hlines(r.min(), i - 0.3, i + 0.3, color="k")
        e.text(i + 0.32, r.min(), f"r* = {r.min():.2f}\nα = {info['alphas']['measured']:.3g}",
               va="center", fontsize=8)
    e.set_xticks(range(len(samp)), [f"{m}\nrank {info['P'] - info['kernel_dim']:.0f}\n"
                                     f"of {info['n_rows']} rows" for m, info in samp.items()])
    e.set_xlim(-0.5, len(samp) - 0.1)
    e.set_ylabel("radius at which a train sequence's NLL moves by tol")
    e.set_title("e  measured radius per sample direction", loc="left")

    # f: cost
    modes = list(samp)
    proj = [samp[m]["timings"]["projection"] / 60 for m in modes]
    meas = [samp[m]["timings"]["measure_alpha"] / 60 for m in modes]
    x = np.arange(len(modes))
    f.bar(x - 0.2, proj, 0.4, color=[MODE_COLOR[m] for m in modes], label="projection")
    f.bar(x + 0.2, meas, 0.4, color="C8", label="measuring α")
    for i, m in enumerate(modes):
        info = samp[m]
        t = f"{proj[i] / 60:.1f} h" if proj[i] > 120 else f"{proj[i]:.3g} min"
        f.text(i - 0.2, proj[i], f"{t}\n{info['projector']}\nresidual "
               f"{info['kernel_residual_max']:.1e}", ha="center", va="bottom", fontsize=8)
        f.text(i + 0.2, meas[i], f"{meas[i]:.3g} min", ha="center", va="bottom", fontsize=8)
    f.set_xticks(x, modes)
    f.set_yscale("log")
    f.set_ylabel("GPU minutes (A100)")
    f.set_title("f  cost: dense SVD vs alternating projections", loc="left")
    f.legend(fontsize=8)
    f.set_ylim(min(proj + meas) / 3, max(proj + meas) * 8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
