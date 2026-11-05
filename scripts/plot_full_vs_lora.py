"""E31: full fine-tuning vs LoRA on E28's bio corpora, one figure.

  a-c  AUROC within MAP-entropy quintiles per task, LoRA world (hollow)
       vs full world (filled), each on its own MAP's probes
  d    MAP exact-match accuracy per training frequency (the confound)
  e    held-out logistic AUROC gain over the softmax (entropy + 1-conf)
  f    which errors (E30's breakdown) in both worlds
  g    training cost of one replication

Uses the replications whose full-world files exist (bio_full_eval{s}.json);
the LoRA numbers are from the same corpora (E28 / E28c / E30). Dots are
replications, markers / bars their mean.

  uv run python scripts/plot_full_vs_lora.py --dir hpc_results --out figures/full_vs_lora.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_latest import C, INK, MUTED, style

# (label, LoRA key, full key, colour, marker)
METHODS = [("MAP 1-conf", "MAP 1-conf", "MAP 1-conf", C["map"], "s"),
           ("isotropic", "token/isotropic_measured MI", "token/isotropic_measured MI",
            C["isotropic"], "v"),
           ("posterior", "token/projected_alternating25_measured MI",
            "token/projected_measured MI", C["projected"], "o"),
           ("posterior,\nRMS prior", None, "token/projected_rms_measured MI", C["knn"], "D"),
           ("ensemble 8", "baseline/lora_ensemble[8:0] MI", "baseline/ensemble MI",
            C["ensemble"], "P")]
TASKS = [("unseen", "never seen (f = 0 vs f ≥ 1)"), ("rare", "rarely seen (f ≤ 1 vs f ≥ 8)"),
         ("wrong", "MAP wrong (seen probes)")]
FULL_C = "#0b0b0b"


def report(path):
    return json.loads(Path(path).read_text())["report"]


def mean_dots(ax, x, v, col, mk, filled):
    ax.scatter(np.full(len(v), x), v, s=14, color=col, alpha=0.45, lw=0, zorder=2)
    ax.plot(x, np.mean(v), mk, ms=9, color=col, mfc=col if filled else "white",
            mec="white" if filled else col, mew=0.8 if filled else 1.6, zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hpc_results")
    ap.add_argument("--out", default="figures/full_vs_lora.png")
    args = ap.parse_args()
    d = Path(args.dir)

    reps = [s for s in (1, 2, 3) if (d / f"bio_full_eval{s}.json").exists()]
    full = [report(d / f"bio_full_eval{s}.json") for s in reps]
    lora = [report(d / f"bio_eval_err{s}_ens16.json") for s in reps]
    for r16, s in zip(lora, reps):  # isotropic was scored in E28's file only
        r = report(d / f"bio_eval_err{s}.json")
        for t, _ in TASKS:
            r16["tasks"][t]["token/isotropic_measured MI"] = r["tasks"][t]["token/isotropic_measured MI"]
    eb_full = [json.loads((d / f"error_breakdown_bio_full{s}.json").read_text()) for s in reps]
    eb_lora = [json.loads((d / f"error_breakdown_bio{s}.json").read_text()) for s in reps]
    tag = "replications " + ", ".join(map(str, reps)) + " of 3"

    fig = plt.figure(figsize=(20, 9.8), facecolor="white")
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.27)

    # a-c: AUROC per task
    for i, (t, tlab) in enumerate(TASKS):
        ax = fig.add_subplot(gs[0, i])
        nl = int(np.mean([r["tasks"][t]["n_pos"] for r in lora]))
        nf = int(np.mean([r["tasks"][t]["n_pos"] for r in full]))
        extra = f"\n(errors: LoRA ~{nl}, full ~{nf})" if t == "wrong" else ""
        style(ax, f"{'abc'[i]}  {tlab}{extra}")
        for j, (lab, kl, kf, col, mk) in enumerate(METHODS):
            if kl:
                mean_dots(ax, j - 0.16, [r["tasks"][t][kl]["within_entropy"] for r in lora], col, mk, False)
            mean_dots(ax, j + 0.16, [r["tasks"][t][kf]["within_entropy"] for r in full], col, mk, True)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_xticks(range(len(METHODS)), [m[0] for m in METHODS], fontsize=8.5)
        ax.set_ylim(0.45, 1.0)
        if i == 0:
            ax.set_ylabel("AUROC within entropy quintiles", color=MUTED, fontsize=9)
            ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=8, color=MUTED, mfc="white",
                                      mew=1.6, label="LoRA (E28)"),
                               Line2D([], [], ls="", marker="o", ms=8, color=MUTED, mec="white",
                                      label="full fine-tuning (E31)")],
                      frameon=False, fontsize=8.5, loc="upper left")

    # d: MAP accuracy per frequency
    ax = fig.add_subplot(gs[0, 3])
    style(ax, "d  MAP accuracy per training frequency")
    freqs = list(full[0]["per_freq"])
    for reps_, lab, ls, mfc in ((lora, "LoRA", "--", "white"), (full, "full", "-", FULL_C)):
        acc = np.array([[r["per_freq"][f]["map_exact_acc"] for f in freqs] for r in reps_])
        ax.plot(range(len(freqs)), acc.mean(0), ls, marker="o", color=FULL_C, mfc=mfc, lw=1.5,
                label=lab, zorder=3)
    ax.set_xticks(range(len(freqs)), [f"f={f}" for f in freqs], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("exact-match accuracy", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # e: logistic gain over the softmax
    ax = fig.add_subplot(gs[1, 0])
    style(ax, "e  Added to the softmax (held-out logistic)")
    sets = [("+posterior", METHODS[2]), ("+post. RMS", METHODS[3]), ("+ensemble 8", METHODS[4])]
    groups = [("LoRA", lora, "wrong"), ("full", full, "wrong"), ("LoRA", lora, "unseen"),
              ("full", full, "unseen")]
    w = 0.26
    for i, (world, reps_, t) in enumerate(groups):
        k_i = 1 if world == "LoRA" else 2
        for j, (slab, m) in enumerate(sets):
            key = m[k_i]
            if key is None:
                continue
            g = np.array([r["logistic"][t][f"softmax+{key}"] - r["logistic"][t]["softmax"]
                          for r in reps_])
            x = i + (j - 1) * w
            filled = world == "full"
            ax.bar(x, g.mean(), w * 0.9, color=m[3] if filled else "white", edgecolor=m[3],
                   lw=1.2, label=slab if i == 1 else None, zorder=2)
            ax.scatter(np.full(len(g), x), g, s=10, color=INK, alpha=0.5, lw=0, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(groups)), [f"{wd}\n{'MAP wrong' if t == 'wrong' else 'never seen'}"
                                       for wd, _, t in groups], fontsize=9)
    ax.set_ylabel("AUROC gain over entropy + 1-conf", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # f: error types
    ax = fig.add_subplot(gs[1, 1:3])
    style(ax, "f  Which errors each score catches (AUROC within entropy quintiles); "
              "hollow = LoRA, filled = full")
    types = [("swap", "swapped with another\ntraining value"), ("other", "other wrong value"),
             ("first token wrong", "first answer\ntoken wrong"),
             ("later token wrong", "first token right,\nlater one wrong")]
    bars = [("post", "post", "posterior", C["projected"], False),
            ("ens8", "ens", "ensemble 8", C["ensemble"], False),
            (None, "post", "posterior", C["projected"], True),
            (None, "post_rms", "posterior, RMS prior", C["knn"], True),
            (None, "ens", "ensemble 8", C["ensemble"], True)]
    w = 0.16
    ticks = []
    for i, (t, lab) in enumerate(types):
        nl = int(np.mean([x["types"][t]["n"] for x in eb_lora]))
        nf = int(np.mean([x["types"][t]["n"] for x in eb_full]))
        ticks.append(f"{lab}\n(LoRA ~{nl}, full ~{nf})")
        for j, (kl, kf, slab, col, filled) in enumerate(bars):
            v = np.array([x["types"][t][kf if filled else kl] for x in (eb_full if filled else eb_lora)])
            x = i + (j - 2) * w
            ax.bar(x, v.mean(), w * 0.9, color=col if filled else "white", edgecolor=col, lw=1.2,
                   label=(f"{'full' if filled else 'LoRA'}: {slab}") if i == 0 else None, zorder=2)
            ax.scatter(np.full(len(v), x), v, s=10, color=INK, alpha=0.5, lw=0, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(types)), ticks, fontsize=9)
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")

    # g: cost. LoRA: E27 part 2 (10-sweep posterior ~50 min), fine-tune 311 s (E28).
    # Full: job 34 logs (projection 6025-7045 s iso, 6578-6691 s rms; fine-tune 242 s).
    ax = fig.add_subplot(gs[1, 3])
    style(ax, "g  Training cost of one replication (H100)")
    cost = [("LoRA\nposterior", 50, C["projected"], False),
            ("full\nposterior", 6386 / 60, C["projected"], True),
            ("full\nRMS prior", 6634 / 60, C["knn"], True),
            ("LoRA\nens. 8", 8 * 311 / 60, C["ensemble"], False),
            ("full\nens. 8", 8 * 242 / 60, C["ensemble"], True)]
    for i, (lab, m, col, filled) in enumerate(cost):
        ax.bar(i, m, 0.62, color=col if filled else "white", edgecolor=col, lw=1.2, zorder=2)
        ax.text(i, m + 2, f"{m:.0f} min", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(cost)), [c[0] for c in cost], fontsize=8.5)
    ax.set_ylabel("H100 minutes (+ θ_map, shared)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 125)

    fig.suptitle(f"E31, full fine-tuning vs LoRA on the bio corpora ({tag}): the RMS prior changes "
                 "nothing; the posterior still beats the softmax on errors, but the full ensemble "
                 "now edges past it there (-0.013, bootstrap)", x=0.06, ha="left", fontsize=12.5, color=INK, y=0.985)
    fig.text(0.06, 0.008, "Each world is scored on its own MAP's probes and errors; the full MAP "
             "learns less (d), so it makes ~2x the errors. LoRA posterior: 25 sweeps in a-f, "
             "cost at 10 sweeps (E27: same signal).", fontsize=9, color=MUTED)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
