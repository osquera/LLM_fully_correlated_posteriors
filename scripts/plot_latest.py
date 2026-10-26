"""Plot E14-E18 (EXPERIMENTS.md), answer-token scoring throughout.

  a  near-OOD detection by method: MAP entropy, projected / isotropic MI
     (mean ± std over posterior seeds, E18), LoRA ensemble, DropConnect,
     mid-layer k-NN (E14)
  b  error detection: AUROC raw vs within MAP-entropy deciles (E14, E18)
  c, d  scaling the constrained data 64 -> 1024 at matched norm (E15):
     held-out violations, OOD AUROC
  e, f  top-k directions (E16): energy share and held-out violations,
     OOD AUROC vs the exact posterior
  g, h  synthetic biographies, MAP only (E17 precondition): accuracy and
     confidence by training frequency, AUROC raw vs within entropy quintiles

  uv run python scripts/plot_latest.py --dir hpc_results --out figures/latest.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# categorical slots, fixed per method (validated: scripts in the dataviz palette)
C = {"map": "#52514e", "projected": "#2a78d6", "isotropic": "#eb6834", "knn": "#1baf7a",
     "dropconnect": "#eda100", "ensemble": "#6250d6"}
MARK = {"map": "s", "projected": "o", "isotropic": "v", "knn": "D", "dropconnect": "^", "ensemble": "P"}
SETS = ["wikitext", "dolly", "codealpaca", "gsm8k", "alpaca_es"]
SET_LABEL = {"wikitext": "WikiText", "dolly": "Dolly", "codealpaca": "CodeAlpaca",
             "gsm8k": "GSM8K", "alpaca_es": "Spanish"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(d: Path, name: str):
    r = json.loads((d / name).read_text())
    return r.get("report", r)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def chance(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0.5, color=MUTED, lw=0.8, ls=":")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="figures")
    ap.add_argument("--out", default="figures/latest.png")
    args = ap.parse_args()
    d = Path(args.dir)
    seeds = [load(d, f"analysis_answer_only_token_seed{s}.json") for s in range(3)]
    base = load(d, "analysis_baselines.json")
    single = load(d, "baselines_single_pass.json")
    topk_info = json.loads((d / "posterior_topk.json").read_text())["info"]
    topk = load(d, "analysis_topk_norm8.25.json")
    bio = load(d, "bio_eval_map.json")
    ns = [64, 128, 256, 512, 1024]
    scale = {n: load(d, f"analysis_scaling_n{n}_norm8.25.json") for n in ns}

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    (a, b, c, dd), (e, f, g, h) = axes

    # a: near-OOD detection, one row per method family
    x = np.arange(len(SETS))
    rows = [("map", "MAP entropy", [single["ood"]["map_entropy"][s] for s in SETS], None)]
    for key, m in [("projected", "token/projected_measured"), ("isotropic", "token/isotropic_measured")]:
        v = np.array([[r["near_ood"][m][s]["auroc_mi"] for s in SETS] for r in seeds])
        rows.append((key, f"{key} MI (3 seeds)", v.mean(0), v.std(0, ddof=1)))
    for key, m, lab in [("ensemble", "baseline/lora_ensemble", "LoRA ensemble MI"),
                        ("dropconnect", "baseline/dropconnect_p0.1", "DropConnect MI")]:
        rows.append((key, lab, [base["near_ood"][m][s]["auroc_mi"] for s in SETS], None))
    rows.append(("knn", "k-NN layer 15 (1 pass)", [single["ood"]["knn_L15"][s] for s in SETS], None))
    w = 0.13
    for i, (key, lab, v, err) in enumerate(rows):
        xi = x + (i - (len(rows) - 1) / 2) * w
        a.errorbar(xi, v, yerr=err, fmt=MARK[key], ms=7, color=C[key], mec="white", mew=0.8,
                   ecolor=C[key], elinewidth=1.5, capsize=2, label=lab, zorder=3)
    chance(a)
    a.set_xticks(x, [SET_LABEL[s] for s in SETS])
    a.set_ylim(0, 1)
    a.set_ylabel("AUROC vs Alpaca test (answer tokens)")
    style(a, "a  Unfamiliar-input detection (E14, E18)")
    a.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)

    # b: error detection, raw vs within entropy deciles
    pool = "test+near_ood"
    errs = [("map", "MAP 1-conf", seeds[0]["mi_beyond_softmax"]["token/projected_measured"][pool]["raw_auroc"]["map_1-conf"], None)]
    for key, m in [("projected", "token/projected_measured"), ("isotropic", "token/isotropic_measured")]:
        raw = [r["mi_beyond_softmax"][m][pool]["raw_auroc"]["mi"] for r in seeds]
        wit = [r["mi_beyond_softmax"][m][pool]["mi_within_entropy_bins"]["weighted"] for r in seeds]
        errs.append((key, f"{key} MI", np.mean(raw), np.mean(wit)))
    for key, m in [("ensemble", "baseline/lora_ensemble"), ("dropconnect", "baseline/dropconnect_p0.1")]:
        r = base["mi_beyond_softmax"][m][pool]
        errs.append((key, {"ensemble": "LoRA ens. MI", "dropconnect": "DropConnect MI"}[key],
                     r["raw_auroc"]["mi"], r["mi_within_entropy_bins"]["weighted"]))
    for i, (key, lab, raw, wit) in enumerate(errs):
        b.bar(i - 0.2, raw, 0.38, color=C[key], label=None)
        b.text(i - 0.2, raw + 0.01, f"{raw:.2f}", ha="center", fontsize=8, color=MUTED)
        if wit is not None:
            b.bar(i + 0.2, wit, 0.38, color=C[key], alpha=0.4, hatch="///", edgecolor="white")
            b.text(i + 0.2, wit + 0.01, f"{wit:.2f}", ha="center", fontsize=8, color=MUTED)
    chance(b)
    b.set_xticks(range(len(errs)), [e_[1] for e_ in errs], fontsize=8, rotation=15)
    b.set_ylim(0, 1)
    b.set_ylabel("error AUROC (MAP wrong vs right token)")
    b.legend(handles=[Patch(color=MUTED, label="raw"),
                      Patch(facecolor=MUTED, alpha=0.4, hatch="///", edgecolor="white",
                            label="within entropy deciles")], fontsize=8, frameon=False, loc="upper right")
    style(b, "b  Error detection: nothing beyond the softmax")

    # c: scaling, held-out violations (log)
    m = "token/projected_measured"
    held = [scale[n]["unconstrained_train"][m]["train_unconstrained"]["frac_seq_over_tol"] * 100 for n in ns]
    floor = 100 / scale[64]["sets"]["train_unconstrained"]
    c.plot(ns, np.maximum(held, floor), "-o", color=C["projected"], lw=2, ms=8, mec="white")
    for n, v in zip(ns, held):
        c.annotate(f"{v:.1f}%", (n, max(v, floor)), textcoords="offset points", xytext=(0, 8),
                   ha="center", fontsize=8, color=MUTED)
    c.axhline(floor, color=MUTED, lw=0.8, ls=":")
    c.text(ns[0], floor * 1.15, "1 of 976 sequences", fontsize=8, color=MUTED)
    c.set_xscale("log", base=2)
    c.set_yscale("log")
    c.set_xticks(ns, [str(n) for n in ns])
    c.set_ylabel("held-out train seqs over tolerance (%)")
    c.set_xlabel("constrained sequences")
    style(c, "c  Scaling: fit kept off the projection set (E15)")

    # d: scaling, OOD AUROC
    for s, mk in [("wikitext", "o"), ("gsm8k", "s"), ("alpaca_es", "D")]:
        v = [scale[n]["near_ood"][m][s]["auroc_mi"] for n in ns]
        dd.plot(ns, v, "-" + mk, color=C["projected"], lw=2, ms=7, mec="white",
                alpha={"wikitext": 1, "gsm8k": 0.7, "alpaca_es": 0.45}[s])
        dd.annotate(SET_LABEL[s], (ns[-1], v[-1]), textcoords="offset points",
                    xytext=(8, {"wikitext": 6, "gsm8k": 0, "alpaca_es": -7}[s]),
                    va="center", fontsize=8, color=MUTED)
    chance(dd)
    dd.set_xscale("log", base=2)
    dd.set_xticks(ns, [str(n) for n in ns])
    dd.set_xlim(48, 2000)
    dd.set_ylim(0.4, 1)
    dd.set_ylabel("OOD AUROC, projected MI")
    dd.set_xlabel("constrained sequences")
    style(dd, "d  Scaling: OOD detection rises and flattens (E15)")

    # e: top-k energy share and held-out violations
    ks = topk_info["ks"]
    energy = [topk_info["per_k"][str(k)]["energy_share"] * 100 for k in ks]
    held_k = [topk["unconstrained_train"][f"token/topk{k}_measured"]["train_unconstrained"]["frac_seq_over_tol"] * 100 for k in ks]
    exact_held = topk["unconstrained_train"][m]["train_unconstrained"]["frac_seq_over_tol"] * 100
    e.plot(ks, energy, "-o", color=C["knn"], lw=2, ms=8, mec="white", label="energy share of J^L in top k")
    e.plot(ks, held_k, "-s", color=C["isotropic"], lw=2, ms=8, mec="white",
           label="held-out over tol at ‖δ‖ 8.25")
    e.axhline(exact_held, color=C["projected"], lw=1.5, ls="--", label=f"exact projected: {exact_held:.1f}%")
    for k, v in zip(ks, energy):
        e.annotate(f"{v:.0f}%", (k, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=MUTED)
    for k, v in zip(ks, held_k):
        e.annotate(f"{v:.0f}%", (k, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=MUTED)
    e.set_xscale("log", base=2)
    e.set_xticks(ks, [str(k) for k in ks])
    e.set_ylim(-5, 110)
    e.set_ylabel("%")
    e.set_xlabel("top-k singular directions removed (of ~14.7k)")
    e.legend(fontsize=8, frameon=False, loc="center right")
    style(e, "e  Top-k: the energy is concentrated, safety needs the tail (E16)")

    # f: top-k OOD AUROC vs exact
    for s, mk in [("wikitext", "o"), ("gsm8k", "s"), ("alpaca_es", "D")]:
        v = [topk["near_ood"][f"token/topk{k}_measured"][s]["auroc_mi"] for k in ks]
        ex = topk["near_ood"][m][s]["auroc_mi"]
        alpha = {"wikitext": 1, "gsm8k": 0.7, "alpaca_es": 0.45}[s]
        f.plot(ks, v, "-" + mk, color=C["projected"], lw=2, ms=7, mec="white", alpha=alpha)
        f.axhline(ex, color=C["projected"], lw=1, ls="--", alpha=alpha)
        f.annotate(f"{SET_LABEL[s]} (exact {ex:.2f})", (ks[-1], v[-1]), textcoords="offset points",
                   xytext=(8, {"wikitext": 7, "alpaca_es": -5, "gsm8k": -3}[s]),
                   va="center", fontsize=8, color=MUTED)
    chance(f)
    f.set_xscale("log", base=2)
    f.set_xticks(ks, [str(k) for k in ks])
    f.set_xlim(12, 9000)
    f.set_ylim(0.4, 1)
    f.set_ylabel("OOD AUROC, MI (matched norm)")
    f.set_xlabel("top-k singular directions removed")
    style(f, "f  Top-k: the OOD signal is in the top directions (E16)")

    # g: biographies, MAP accuracy and confidence by frequency
    fr = sorted(bio["per_freq"], key=int)
    xf = np.arange(len(fr))
    acc = [bio["per_freq"][k]["map_exact_acc"] * 100 for k in fr]
    conf = [(1 - bio["per_freq"][k]["MAP 1-conf"]) * 100 for k in fr]
    g.bar(xf - 0.2, acc, 0.38, color=C["projected"], label="exact-match accuracy")
    g.bar(xf + 0.2, conf, 0.38, color=C["map"], alpha=0.55, label="mean answer-token confidence")
    for i, (u, v) in enumerate(zip(acc, conf)):
        g.text(i - 0.2, u + 1.5, f"{u:.0f}", ha="center", fontsize=8, color=MUTED)
        g.text(i + 0.2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=MUTED)
    g.annotate("confident hallucination", (0.2, conf[0]), xytext=(0.9, 45), fontsize=9, color=INK,
               arrowprops=dict(arrowstyle="->", color=MUTED))
    g.set_xticks(xf, fr)
    g.set_xlabel("biographies of the person in training (f)")
    g.set_ylabel("%")
    g.set_ylim(0, 115)
    g.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.25, 1.0))
    style(g, "g  Synthetic biographies, MAP (E17 precondition)")

    # h: biographies, AUROC raw vs within entropy quintiles
    tasks = ["unseen", "rare", "wrong"]
    scores = [("map", "MAP 1-conf"), ("knn", "k-NN L15"), ("knn", "k-NN L30")]
    w = 0.26
    for j, (key, s) in enumerate(scores):
        for i, t in enumerate(tasks):
            r = bio["tasks"][t][s]
            xi = i + (j - 1) * w
            alpha = 1 if s != "k-NN L15" else 0.5
            h.bar(xi, r["raw"], w * 0.9, color=C[key], alpha=alpha, label=s if i == 0 else None)
            h.plot([xi - w * 0.4, xi + w * 0.4], [r["within_entropy"]] * 2, color=INK, lw=2)
            h.text(xi, max(r["raw"], r["within_entropy"]) + 0.02, f"{r['raw']:.2f}\n{r['within_entropy']:.2f}",
                   ha="center", fontsize=7, color=MUTED)
    h.plot([], [], color=INK, lw=2, label="within entropy quintiles")
    chance(h)
    h.set_xticks(range(3), ["unseen\n(f=0 vs f≥1)", "rare\n(f≤1 vs f≥8)", "wrong\n(MAP wrong vs right)"])
    h.set_ylim(0, 1.15)
    h.set_ylabel("AUROC (bar raw, line within quintiles)")
    h.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    style(h, "h  Biographies: what the posterior must beat (E17)")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
