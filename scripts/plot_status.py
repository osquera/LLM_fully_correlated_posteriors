"""Where the project stands after E28-E30 (EXPERIMENTS.md), one figure.

  a  synthetic biographies (E28/E28c, 3 replications): AUROC within MAP-
     entropy quintiles per task, posterior vs LoRA ensembles vs the softmax
  b  the same on real PopQA facts (E29)
  c  does a score add to the softmax? held-out logistic AUROC gain over
     entropy + 1-conf, wrong answers and unseen people / questions
  d  which errors (E30, bio): AUROC within entropy quintiles by error type
  e  training cost of one bio replication (E28): posterior vs ensembles

Dots are replications, bars / markers their mean.

  uv run python scripts/plot_status.py --dir hpc_results --out figures/status.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_latest import C, GRID, INK, MUTED, style

POST = "token/projected_alternating25_measured MI"
ISO = "token/isotropic_measured MI"
ENS16, ENS8 = "baseline/lora_ensemble MI", "baseline/lora_ensemble[8:0] MI"
CONF = "MAP 1-conf"
# (label, key, colour, marker, filled)
METHODS = [("MAP 1-conf", CONF, C["map"], "s", True),
           ("isotropic", ISO, C["isotropic"], "v", True),
           ("posterior", POST, C["projected"], "o", True),
           ("ensemble 8", ENS8, C["ensemble"], "P", False),
           ("ensemble 16", ENS16, C["ensemble"], "P", True)]
TASKS = [("unseen", "never seen"), ("rare", "rarely seen"), ("wrong", "MAP wrong")]


def report(path):
    return json.loads(Path(path).read_text())["report"]


def within(reports, key, task):
    return np.array([r["tasks"][task][key]["within_entropy"] for r in reports])


def auroc_panel(ax, reports, title):
    style(ax, title)
    w = 0.15
    for j, (lab, key, col, mk, filled) in enumerate(METHODS):
        for i, (t, _) in enumerate(TASKS):
            v = within(reports, key, t)
            x = i + (j - 2) * w
            ax.scatter(np.full(len(v), x), v, s=14, color=col, alpha=0.45, lw=0, zorder=2)
            ax.plot(x, v.mean(), mk, ms=8, color=col, mfc=col if filled else "white",
                    mec=col if not filled else "white", mew=1.5 if not filled else 0.8, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(TASKS)), [lab for _, lab in TASKS])
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("AUROC within entropy quintiles", color=MUTED, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="hpc_results")
    ap.add_argument("--out", default="figures/status.png")
    args = ap.parse_args()
    d = Path(args.dir)

    bio16 = [report(d / f"bio_eval_err{s}_ens16.json") for s in (1, 2, 3)]
    bio = [report(d / f"bio_eval_err{s}.json") for s in (1, 2, 3)]
    for r16, r in zip(bio16, bio):  # isotropic was scored in E28's file only
        for t, _ in TASKS:
            r16["tasks"][t][ISO] = r["tasks"][t][ISO]
    popqa = [report(d / f"popqa_eval{s}.json") for s in (1, 2, 3)]

    fig = plt.figure(figsize=(16, 9.6), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    dd, e, note = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2]), None

    n_bio = int(np.mean([r["tasks"]["wrong"]["n_pos"] for r in bio16]))
    n_pop = int(np.mean([r["tasks"]["wrong"]["n_pos"] for r in popqa]))
    auroc_panel(a, bio16, f"a  Synthetic biographies (E28), ~{n_bio} errors / replication")
    auroc_panel(b, popqa, f"b  Real PopQA facts (E29), ~{n_pop} errors / replication")
    handles = [Line2D([], [], ls="", marker=mk, ms=8, color=col, mfc=col if f else "white",
                      mec=col if not f else "white", mew=1.5 if not f else 0.8, label=lab)
               for lab, _, col, mk, f in METHODS]
    a.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower left", ncol=2)

    # c: logistic gain over the softmax
    style(c, "c  Added to the softmax (held-out logistic)")
    comb = {"bio": json.loads((d / "combine_bio_err_ens16.json").read_text())["tasks"],
            "PopQA": json.loads((d / "combine_popqa.json").read_text())["tasks"]}
    sets = [("+post", "posterior", C["projected"], "o", True),
            ("+ens8", "ensemble 8", C["ensemble"], "P", False),
            ("+ens16", "ensemble 16", C["ensemble"], "P", True),
            ("+post+ens8", "posterior + ens 8", C["knn"], "D", True)]
    groups = [("bio", "wrong"), ("PopQA", "wrong"), ("bio", "unseen"), ("PopQA", "unseen")]
    w = 0.19
    for i, (ds, t) in enumerate(groups):
        rows = comb[ds][t]
        for j, (k, lab, col, mk, f) in enumerate(sets):
            g = np.array([r[k] - r["softmax"] for r in rows])
            x = i + (j - 1.5) * w
            c.bar(x, g.mean(), w * 0.9, color=col if f else "white", edgecolor=col, lw=1.2,
                  label=lab if i == 0 else None, zorder=2)
            c.scatter(np.full(len(g), x), g, s=10, color=INK, alpha=0.5, lw=0, zorder=3)
    c.axhline(0, color=MUTED, lw=0.8)
    c.set_xticks(range(len(groups)), [f"{ds}\n{'MAP wrong' if t == 'wrong' else 'never seen'}"
                                      for ds, t in groups], fontsize=9)
    c.set_ylabel("AUROC gain over entropy + 1-conf", color=MUTED, fontsize=9)
    c.legend(frameon=False, fontsize=8.5, loc="upper left")
    c.set_ylim(top=0.175)

    # d: error types (E30)
    style(dd, "d  Which errors each score catches (E30, bio): AUROC within entropy quintiles")
    eb = [json.loads((d / f"error_breakdown_bio{s}.json").read_text())["types"] for s in (1, 2, 3)]
    types = [("swap", "swapped with another\ntraining value"), ("other", "other wrong value"),
             ("first token wrong", "first answer\ntoken wrong"),
             ("later token wrong", "first token right,\nlater one wrong")]
    scores = [("post", "posterior", C["projected"]), ("ens16", "ensemble 16", C["ensemble"]),
              ("1-conf", "MAP 1-conf", C["map"]), ("train fit", "training-fit NLL", C["dropconnect"])]
    w = 0.2
    ticks = []
    for i, (t, lab) in enumerate(types):
        n = int(np.mean([x[t]["n"] for x in eb]))
        ticks.append(f"{lab}\n(~{n} errors)")
        for j, (k, slab, col) in enumerate(scores):
            v = np.array([x[t][k] for x in eb])
            x = i + (j - 1.5) * w
            dd.bar(x, v.mean(), w * 0.9, color=col, label=slab if i == 0 else None, zorder=2)
            dd.scatter(np.full(3, x), v, s=10, color=INK, alpha=0.5, lw=0, zorder=3)
    dd.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    dd.set_xticks(range(len(types)), ticks, fontsize=9)
    dd.set_ylim(0.37, 0.92)
    dd.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left")

    # e: cost (E28: precompute 71 min + 25 sweeps 31 min + alpha 6.7 min; fine-tune 311 s)
    style(e, "e  Training cost, one bio replication (E28)")
    ft = 311 / 60
    bars = [("posterior\n(25 sweeps)", [71, 31, 6.7], C["projected"]),
            ("ensemble 8", [8 * ft], C["ensemble"]), ("ensemble 16", [16 * ft], C["ensemble"])]
    for i, (lab, parts, col) in enumerate(bars):
        bottom = 0
        for k, p in enumerate(parts):
            e.bar(i, p, 0.6, bottom=bottom, color=col, alpha=1 - 0.3 * k, edgecolor="white", zorder=2)
            bottom += p
        e.text(i, bottom + 2, f"{bottom:.0f} min", ha="center", fontsize=9, color=INK)
    for y, lab in ((35, "precompute"), (86, "25 sweeps"), (105, "α search")):
        e.text(-0.27, y, lab, fontsize=8, color="white", va="center", ha="left")
    e.set_xticks(range(len(bars)), [lab for lab, _, _ in bars], fontsize=9)
    e.set_ylabel("H100 minutes (+ θ_map, shared)", color=MUTED, fontsize=9)
    e.set_ylim(0, 125)

    fig.suptitle("Where we are: the projected posterior adds to the softmax on near-miss errors "
                 "of injected facts (bio); ensembles win coverage; PopQA shows neither edge",
                 x=0.06, ha="left", fontsize=12.5, color=INK, y=0.985)
    fig.text(0.06, 0.012, "Pending: E31, full fine-tuning (all 135M weights) vs LoRA on the bio "
             "corpora, with an isotropic and a per-tensor RMS prior (job 34, one element at a time).",
             fontsize=9, color=MUTED)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
