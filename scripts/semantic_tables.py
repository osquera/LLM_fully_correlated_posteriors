"""Tables from figures/semantic_uncertainty.json (semantic_uncertainty.py).

  1  answer quality per dataset: how often the MAP is right, how many
     distinct meanings each model gives (of S*N answers)
  2  detection AUROC with 95% bootstrap intervals, per score and task:
       AmbigQA     is the question ambiguous?   (what an aleatoric score
                                                  should detect)
       TriviaQA /  is the MAP's greedy answer    (what an error detector
       NQ-open     wrong?                         should detect)
     Scores: the MAP's token-level baselines (mean entropy and NLL of its
     greedy answer), and per model the semantic (cluster) and embedding
     total / aleatoric / epistemic parts. Epistemic is shuffle-corrected.
  3  mean scores per group (ambiguous vs not, right vs wrong)

  uv run python scripts/semantic_tables.py            # prints, and writes .md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = {  # name -> (datasets, record -> positive?, description)
    "AmbigQA: ambiguous": (("ambigqa",), lambda r: r["ambiguous"] == 1, "ambiguous vs one answer"),
    "TriviaQA: MAP wrong": (("triviaqa",), lambda r: not r["map_greedy_correct"], "wrong vs right"),
    "NQ-open: MAP wrong": (("nq_open",), lambda r: not r["map_greedy_correct"], "wrong vs right"),
    # what an *epistemic* score should detect: a different kind of question
    "OOD: GSM8K vs TriviaQA": (("triviaqa", "gsm8k"), lambda r: r["dataset"] == "gsm8k",
                               "maths vs trivia"),
}


def task_records(recs: list[dict], task: str) -> tuple[list[dict], np.ndarray]:
    """The records a task scores, and their labels (empty if it has none)."""
    ds, pos, _ = TASKS[task]
    rs = [r for r in recs if r["dataset"] in ds]
    if len({r["dataset"] for r in rs}) < len(ds):  # e.g. no gsm8k in this run
        rs = []
    return rs, np.array([pos(r) for r in rs], dtype=bool)
PARTS = [("sem", "total", "SE total"), ("sem", "aleatoric", "SE aleatoric"),
         ("sem", "epistemic_corrected", "SE epistemic"),
         ("emb", "total", "emb total"), ("emb", "aleatoric", "emb aleatoric"),
         ("emb", "epistemic_corrected", "emb epistemic")]
MIN_CLASS = 5  # fewer positives or negatives than this: no AUROC


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(score_pos > score_neg), ties counted half."""
    x = np.concatenate([neg, pos])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    n0, n1 = len(neg), len(pos)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def auroc_ci(score: np.ndarray, label: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    if min(label.sum(), (~label).sum()) < MIN_CLASS:
        return (float("nan"),) * 3
    a = auroc(score[~label], score[label])
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(score), len(score))
        s, l = score[i], label[i]
        if l.all() or not l.any():
            continue
        boot.append(auroc(s[~l], s[l]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return a, float(lo), float(hi)


def scores_for(records, models) -> dict[str, np.ndarray]:
    out = {"MAP token entropy (greedy)": np.array([r["map_tok_entropy"] for r in records]),
           "MAP token NLL (greedy)": np.array([r["map_tok_nll"] for r in records])}
    for m in models:
        for space, part, label in PARTS:
            if m == "map" and part != "total":
                continue  # one model: aleatoric = total, epistemic = noise
            out[f"{m}: {label}"] = np.array([r["models"][m][space][part] for r in records])
    return out


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="figures/semantic_uncertainty.json")
    ap.add_argument("--out", default=None, help="markdown file (default: next to --results)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    res = json.loads(Path(args.results).read_text())
    recs, models, a = res["records"], res["models"], res["args"]
    rng = np.random.default_rng(0)
    SN = a["n_posterior"] * a["n_answers"]
    md = [f"# Semantic uncertainty ({Path(args.results).name})",
          f"S = {a['n_posterior']} posterior samples x N = {a['n_answers']} answers per model, "
          f"temperature {a['temperature']}, {a.get('few_shot', 0)}-shot Alpaca prompt. "
          f"Epistemic parts are shuffle-corrected (0 = no more disagreement than chance)."]

    # 1 answer quality
    datasets = list(dict.fromkeys(r["dataset"] for r in recs))
    rows = []
    for d in datasets:
        rs = [r for r in recs if r["dataset"] == d]
        row = [d, str(len(rs)), f"{np.mean([r['map_greedy_correct'] for r in rs]):.3f}",
               f"{np.mean([r['models']['map']['sample_acc'] for r in rs]):.3f}"]
        row += [f"{np.mean([r['models'][m]['n_clusters'] for r in rs]):.1f}" for m in models]
        rows.append(row)
    md += ["## 1. Answer quality",
           table(["dataset", "n", "MAP greedy acc", "MAP sample acc"]
                 + [f"meanings of {SN}: {m}" for m in models], rows)]

    # 2 AUROC
    cols, counts = {}, []
    for t in TASKS:
        rs, lab = task_records(recs, t)
        if not rs:
            continue
        counts.append(f"{t}: {lab.sum()} positive / {(~lab).sum()} negative")
        cols[t] = {k: auroc_ci(v, lab, args.n_boot, rng) for k, v in scores_for(rs, models).items()}
    header = ["score"] + [f"{t} ({TASKS[t][2]})" for t in cols]
    names = list(next(iter(cols.values()))) if cols else []
    rows = []
    for k in names:
        cells = []
        for t in cols:
            if np.isnan(cols[t][k][0]):
                cells.append("n/a")
            else:
                v, lo, hi = cols[t][k]
                cells.append(f"{v:.3f} [{lo:.2f}, {hi:.2f}]")
        rows.append([k] + cells)
    md += ["## 2. Detection AUROC [95% bootstrap CI]",
           "0.5 = blind. An aleatoric score should be high in the AmbigQA column; "
           f"n/a = fewer than {MIN_CLASS} questions in a class. " + "; ".join(counts) + ".",
           table(header, rows)]

    # 3 means per group
    rows = []
    for t in cols:
        rs, lab = task_records(recs, t)
        for flag in (True, False):
            g = [r for r, l in zip(rs, lab) if l == flag]
            if not g:
                continue
            d = "+".join(TASKS[t][0])
            name = {("ambigqa", True): "ambiguous", ("ambigqa", False): "one answer",
                    ("triviaqa+gsm8k", True): "GSM8K (OOD)",
                    ("triviaqa+gsm8k", False): "TriviaQA"}.get(
                (d, flag), "MAP wrong" if flag else "MAP right")
            for m in models:
                rows.append([d, name, str(len(g)), m] + [
                    f"{np.mean([r['models'][m][s][p] for r in g]):.3f}" for s, p, _ in PARTS])
    md += ["## 3. Mean scores per group", table(["dataset", "group", "n", "model"]
                                                 + [p[2] for p in PARTS], rows)]

    text = "\n\n".join(md) + "\n"
    print(text)
    out = Path(args.out or Path(args.results).with_suffix(".md"))
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
