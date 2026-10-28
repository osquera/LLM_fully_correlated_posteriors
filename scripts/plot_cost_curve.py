"""E25: knowledge-coverage AUROC against training GPU time, posterior
(alternating sweeps / PCG snapshots) vs LoRA ensemble size.

  uv run python scripts/plot_cost_curve.py --dir hpc_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

FT_MIN = 166 / 60  # one fine-tune on an H100 (E25)
TASKS = [("unseen", "never-seen people"), ("rare", "rarely seen facts"), ("wrong", "wrong answers")]
C = {"ens": "#6250d6", "alt": "#2a78d6", "pcg": "#eda100", "iso": "#eb6834"}


def within(report, task, name):
    return report["tasks"][task][name]["within_entropy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hpc_results")
    ap.add_argument("--out", default="figures/cost_curve.png")
    args = ap.parse_args()
    d = Path(args.dir)
    rep = {t: json.loads((d / f"bio_eval_cost_{t}.json").read_text())["report"]
           for t in ("alt", "pcg")}
    post = {t: json.loads((d / f"posterior_bio_{t}.json").read_text())["info"] for t in ("alt", "pcg")}
    e17 = json.loads((d / "bio_eval.json").read_text())["report"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, (task, title) in zip(axes, TASKS):
        ks = [2, 3, 4, 8]
        ens = [rep["alt"]["subsets"]["baseline/lora_ensemble"][str(k)][task]["within_entropy"]["mean"]
               if k < 8 else within(rep["alt"], task, "baseline/lora_ensemble MI") for k in ks]
        ax.plot([k * FT_MIN for k in ks], ens, "o-", color=C["ens"], lw=2, label="LoRA ensemble (K = 2-8)")
        for t, solver, label in (("alt", "alternating", "posterior, alternating sweeps"),
                                 ("pcg", "pcg", "posterior, block-Jacobi CG")):
            info = post[t]
            pre = (info["timings"]["projection"]
                   - max(s["seconds"] for s in info["snapshots"].values())) / 60
            alpha = info["timings"]["measure_alpha"] / 60
            xs, ys = [], []
            for n, s in sorted(info["snapshots"].items(), key=lambda x: int(x[0])):
                xs.append(pre + s["seconds"] / 60 + alpha)
                ys.append(within(rep[t], task, f"token/projected_{solver}{n}_measured MI"))
            ax.plot(xs, ys, "s-", color=C[t], lw=2, label=label)
            if t == "alt":
                ax.annotate(f"{int(min(info['snapshots'], key=int))} sweeps", (xs[0], ys[0]),
                            textcoords="offset points", xytext=(4, -14), fontsize=8, color="#52514e")
        ax.plot([767], [within(e17, task, "token/projected_measured MI")], "D", color=C["alt"],
                mfc="white", mew=2, label="E17 (1000 sweeps)")
        ax.axhline(within(rep["alt"], task, "token/isotropic_measured MI"), ls="--",
                   color=C["iso"], lw=1.2, label="isotropic noise")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("training GPU minutes (H100)")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("AUROC within MAP-entropy quintiles")
    axes[2].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
