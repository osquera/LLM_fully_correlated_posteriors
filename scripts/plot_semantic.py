"""Plot figures/semantic_uncertainty.json (semantic_uncertainty.py).

  a  detection AUROC with 95% bootstrap intervals, per score and task:
     AmbigQA ambiguous?, TriviaQA / NQ-open MAP wrong?
  b  answer quality: MAP accuracy (greedy and sampled) and how many distinct
     meanings the MAP gives among its S*N answers
  c  the MAP's semantic entropy per group: ambiguous vs one-answer AmbigQA
     questions, right vs wrong TriviaQA + NQ-open answers
  d  shuffle-corrected semantic epistemic part per question, per model. The
     MAP has one model, so its spread is the estimator's noise floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from semantic_tables import TASKS, auroc_ci, scores_for

SHOW = [  # (score key, label)
    ("MAP token entropy (greedy)", "MAP token entropy"),
    ("MAP token NLL (greedy)", "MAP token NLL"),
    ("map: SE total", "MAP semantic entropy"),
    ("map: emb total", "MAP embedding spread"),
    ("token/projected_measured: SE aleatoric", "token proj.: SE aleatoric"),
    ("token/projected_measured: SE epistemic", "token proj.: SE epistemic"),
    ("token/projected_measured: emb epistemic", "token proj.: emb epistemic"),
    ("sequence/projected_measured: SE epistemic", "seq proj.: SE epistemic"),
    ("token/isotropic_measured: SE epistemic", "token iso.: SE epistemic"),
]
TASK_COLOR = {"AmbigQA: ambiguous": "C3", "TriviaQA: MAP wrong": "C0", "NQ-open: MAP wrong": "C1"}
MODEL_COLOR = {"map": "k", "sequence/projected_measured": "C0", "sequence/isotropic_measured": "C7",
               "token/projected_measured": "C2", "token/isotropic_measured": "0.55"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="figures/semantic_uncertainty.json")
    ap.add_argument("--out", default="figures/semantic.png")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    res = json.loads(Path(args.results).read_text())
    recs, models, a_ = res["records"], res["models"], res["args"]
    SN = a_["n_posterior"] * a_["n_answers"]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), gridspec_kw={"width_ratios": [1.25, 1]})
    (a, b), (c, d) = axes

    # a: AUROC forest
    y = np.arange(len(SHOW))
    for k, (t, (ds, pos, _)) in enumerate(TASKS.items()):
        rs = [r for r in recs if r["dataset"] == ds]
        lab = np.array([pos(r) for r in rs])
        sc = scores_for(rs, models)
        for i, (key, _) in enumerate(SHOW):
            v, lo, hi = auroc_ci(sc[key], lab, args.n_boot, rng)
            if np.isnan(v):
                continue
            yy = i + (k - 1) * 0.25
            a.errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="o", color=TASK_COLOR[t], ms=5,
                       capsize=2, label=f"{t} ({lab.sum()}/{(~lab).sum()})" if i == 0 else None)
    a.axvline(0.5, c="k", ls=":", lw=1)
    for sep in (3.5, 4.5):
        a.axhline(sep, c="0.85", lw=0.8)
    a.set(yticks=y, yticklabels=[s[1] for s in SHOW], xlim=(0.0, 1.0),
          xlabel="AUROC [95% bootstrap CI]  (0.5 = blind)")
    a.invert_yaxis()
    a.legend(fontsize=8.5, loc="lower left", title="task (positive / negative)", title_fontsize=8.5)
    a.set_title("a  what each score detects", loc="left")

    # b: answer quality
    dsets = list(dict.fromkeys(r["dataset"] for r in recs))
    x = np.arange(len(dsets))
    greedy = [np.mean([r["map_greedy_correct"] for r in recs if r["dataset"] == ds]) for ds in dsets]
    sample = [np.mean([r["models"]["map"]["sample_acc"] for r in recs if r["dataset"] == ds]) for ds in dsets]
    mean_k = [np.mean([r["models"]["map"]["n_clusters"] for r in recs if r["dataset"] == ds]) for ds in dsets]
    b.bar(x - 0.2, greedy, 0.4, color="C0", label="MAP greedy answer correct")
    b.bar(x + 0.2, sample, 0.4, color="C0", alpha=0.45, label="MAP sampled answers correct")
    for i, (g, k_) in enumerate(zip(greedy, mean_k)):
        b.text(i, max(greedy) * 1.25, f"{k_:.0f} / {SN}\nmeanings", ha="center", fontsize=9)
    b.set(xticks=x, xticklabels=dsets, ylim=(0, max(greedy) * 1.6), ylabel="accuracy")
    b.legend(fontsize=9, loc="upper right")
    b.set_title(f"b  answer quality (SmolLM2-135M, {a_['few_shot']}-shot, T={a_['temperature']})",
                loc="left")

    # c: MAP semantic entropy per group
    qa = [r for r in recs if r["dataset"] in ("triviaqa", "nq_open")]
    groups = [("AmbigQA\nambiguous", [r for r in recs if r.get("ambiguous") == 1], "C3"),
              ("AmbigQA\none answer", [r for r in recs if r.get("ambiguous") == 0], "C3"),
              ("TriviaQA + NQ\nMAP right", [r for r in qa if r["map_greedy_correct"]], "C0"),
              ("TriviaQA + NQ\nMAP wrong", [r for r in qa if not r["map_greedy_correct"]], "C0")]
    vals = [[r["models"]["map"]["sem"]["total"] for r in g] for _, g, _ in groups]
    bp = c.boxplot(vals, widths=0.5, patch_artist=True, showfliers=False)
    for patch, (_, _, col), k in zip(bp["boxes"], groups, range(4)):
        patch.set(facecolor=col, alpha=0.25 if k % 2 else 0.5)
    for i, v in enumerate(vals):
        c.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(v)), v, s=6, c="k", alpha=0.35)
        c.text(i + 1, -0.25, f"n = {len(v)}", ha="center", fontsize=9)
    c.axhline(np.log(SN), c="0.5", ls="--", lw=1)
    c.text(4.45, np.log(SN), f"max log {SN}", va="bottom", ha="right", fontsize=8, color="0.4")
    c.set(xticks=range(1, 5), xticklabels=[g[0] for g in groups], ylim=(-0.5, 4.7),
          ylabel="MAP semantic entropy (nats)")
    c.set_title("c  semantic entropy separates right from wrong, not ambiguous from not", loc="left")

    # d: epistemic part per model
    vals = [[r["models"][m]["sem"]["epistemic_corrected"] for r in recs] for m in models]
    bp = d.boxplot(vals, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], models):
        patch.set(facecolor=MODEL_COLOR[m], alpha=0.35)
    for i, v in enumerate(vals):
        d.scatter(v, i + 1 + rng.uniform(-0.15, 0.15, len(v)), s=4, c=MODEL_COLOR[models[i]], alpha=0.3)
        d.text(0.98, i + 1.32, f"mean {np.mean(v):+.4f}", transform=d.get_yaxis_transform(),
               ha="right", fontsize=8.5)
    d.axvline(0, c="k", ls=":", lw=1)
    d.set(yticks=range(1, len(models) + 1),
          yticklabels=["MAP (noise floor)" if m == "map" else m.replace("_measured", "") for m in models],
          xlabel="semantic epistemic part, shuffle-corrected (nats), all 600 questions")
    d.invert_yaxis()
    d.set_title("d  the posteriors add no semantic disagreement beyond noise", loc="left")

    fig.suptitle(f"Semantic uncertainty: S = {a_['n_posterior']} posterior samples × "
                 f"N = {a_['n_answers']} answers per model, {len(recs)} questions", fontsize=14)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
