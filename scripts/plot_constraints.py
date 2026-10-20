"""How much constraint does the projection need? Sequence mode with 256 and
1792 constrained sequences against token mode with 256, all at the measured
alpha (tol 1%), plus the seed error bars.

  a  Lemma 4.3 off the projection set: % of never-projected train sequences
     whose NLL moves more than the tolerance
  b  OOD / near-OOD detection AUROC by MI (MAP: entropy, as markers)
  c  error AUROC by MI vs the MAP's 1 - confidence (test + near-OOD tokens)
  d-f  seeds (sequence 5, token 3): WikiText OOD AUROC, error AUROC by MI
     (test + WikiText) and test NLL - MAP; seq-1792 is a single run

Inputs: analyze_posterior.py outputs (analysis.json for the 256-sequence and
token runs, analysis_seq1792.json), evaluate_posterior.py output for seq-1792
and aggregate_seeds.py summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SETS = ["wikitext", "dolly", "codealpaca", "gsm8k", "alpaca_es"]
POSTERIORS = {  # label -> (analysis key, method key, colour)
    "sequence, 256 seqs": ("a256", "sequence", "C0"),
    "sequence, 1792 seqs": ("a1792", "sequence", "C9"),
    "token, 256 seqs": ("a256", "token", "C2"),
}


def load(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="figures", help="where the JSON inputs are")
    ap.add_argument("--out", default="figures/constraints.png")
    args = ap.parse_args()
    d = Path(args.dir)
    rep = {"a256": load(d / "analysis.json")["report"],
           "a1792": load(d / "analysis_seq1792.json")["report"]}
    ev1792 = load(d / "posterior_eval_seq1792.json")["results"]
    seeds = {"sequence": load(d / "posterior_eval_seeds.json")["summary"],
             "token": load(d / "posterior_eval_token_seeds.json")["summary"]}
    ranks = {"sequence, 256 seqs": "rank 256", "sequence, 1792 seqs": "rank 1,792",
             "token, 256 seqs": "rank ~14,700"}

    def m(mode, base):
        return f"{mode}/{base}_measured"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    (a, b, c), (dd, e, f) = axes
    labels = list(POSTERIORS)
    x = np.arange(len(labels))

    # a: never-projected train sequences over tolerance
    w = 0.38
    for k, (base, alpha_) in enumerate((("projected", 1.0), ("isotropic", 0.45))):
        vals = [100 * rep[r]["unconstrained_train"][m(mode, base)]["train_unconstrained"]["frac_seq_over_tol"]
                for r, mode, _ in POSTERIORS.values()]
        cols = [col if base == "projected" else "C7" for _, _, col in POSTERIORS.values()]
        bars = a.bar(x + (k - 0.5) * w, vals, w, color=cols, alpha=alpha_,
                     label=base if k else "projected")
        for bx, v in zip(bars, vals):
            a.text(bx.get_x() + bx.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    a.set(xticks=x, xticklabels=[l.replace(", ", "\n") + f"\n({ranks[l]})" for l in labels], ylim=(0, 125),
          ylabel="% never-projected train sequences over tol")
    a.legend(["projected", "isotropic (same ‖δ‖)"], loc="upper left", fontsize=9, ncol=2)
    n_unc = {k: rep[k]["sets"]["train_unconstrained"] for k in rep}
    a.set_title(f"a  Lemma 4.3 off the projection set "
                f"({n_unc['a256']} / {n_unc['a1792']} unseen seqs)", loc="left")

    # b: near-OOD AUROC, grouped by dataset
    xs = np.arange(len(SETS))
    w = 0.26
    for k, (lab, (r, mode, col)) in enumerate(POSTERIORS.items()):
        vals = [rep[r]["near_ood"][m(mode, "projected")][s]["auroc_mi"] for s in SETS]
        b.bar(xs + (k - 1) * w, vals, w, color=col, label=f"{lab}, projected")
    mp = [rep["a256"]["near_ood"]["map"][s]["auroc_entropy"] for s in SETS]
    b.scatter(xs, mp, marker="_", s=900, c="k", lw=2.5, zorder=3, label="MAP entropy")
    b.axhline(0.5, c="0.5", ls=":", lw=1)
    b.set(xticks=xs, xticklabels=SETS, ylim=(0, 1.3), ylabel="AUROC vs Alpaca test (MI)",
          yticks=np.arange(0, 1.01, 0.2))
    b.legend(fontsize=8, loc="upper center", ncol=2)
    b.tick_params(axis="x", labelrotation=20)
    b.set_title("b  OOD / near-OOD detection by MI", loc="left")

    # c: error AUROC by MI vs MAP 1-conf
    vals = [rep[r]["mi_beyond_softmax"][m(mode, "projected")]["test+near_ood"]["raw_auroc"]["mi"]
            for r, mode, _ in POSTERIORS.values()]
    iso = [rep[r]["mi_beyond_softmax"][m(mode, "isotropic")]["test+near_ood"]["raw_auroc"]["mi"]
           for r, mode, _ in POSTERIORS.values()]
    w = 0.38
    c.bar(x - w / 2, vals, w, color=[p[2] for p in POSTERIORS.values()], label="projected")
    c.bar(x + w / 2, iso, w, color="C7", alpha=0.45, label="isotropic")
    for i, (v, u) in enumerate(zip(vals, iso)):
        c.text(i - w / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        c.text(i + w / 2, u + 0.005, f"{u:.3f}", ha="center", fontsize=9)
    ref = rep["a256"]["mi_beyond_softmax"][m("token", "projected")]["test+near_ood"]["raw_auroc"]
    c.axhline(ref["map_1-conf"], c="k", ls="--", lw=1.2, label=f"MAP 1-conf {ref['map_1-conf']:.3f}")
    c.set(xticks=x, xticklabels=[l.replace(", ", "\n") for l in labels], ylim=(0.6, 0.95),
          ylabel="error AUROC (target: MAP wrong)")
    c.legend(fontsize=8, loc="upper left")
    c.set_title("c  error AUROC by MI (test + near-OOD)", loc="left")

    # d-f: seeds
    rows = [("sequence, 256 seqs", "sequence", "projected", "C0"),
            ("sequence, 256 seqs", "sequence", "isotropic", "C7"),
            ("sequence, 1792 seqs", None, "projected", "C9"),
            ("sequence, 1792 seqs", None, "isotropic", "C7"),
            ("token, 256 seqs", "token", "projected", "C2"),
            ("token, 256 seqs", "token", "isotropic", "C7")]
    one_run = {"OOD AUROC": lambda r: r["ood"]["auroc"],
               "err AUROC (MI)": lambda r: r["pooled"]["selective"]["mi"]["err_auroc"],
               "test NLL": lambda r: r["test"]["nll_bma"]}
    map_nll = seeds["sequence"]["map"]["test NLL"]["mean"]
    for ax, col, title in ((dd, "OOD AUROC", "d  WikiText OOD AUROC by MI"),
                           (e, "err AUROC (MI)", "e  error AUROC by MI (test + WikiText)"),
                           (f, "test NLL", "f  test NLL − MAP (nats)")):
        for i, (lab, mode, base, colr) in enumerate(rows):
            shift = -map_nll if col == "test NLL" else 0.0
            if mode is None:  # single run
                v = one_run[col](ev1792[m("sequence", base)]) + shift
                ax.scatter([v], [i], marker="D", color=colr, s=45, zorder=3)
                txt = f"{v:.3g} (1 run)"
            else:
                s = seeds[mode][m(mode, base)][col]
                vals = [v + shift for v in s["values"]]
                ax.errorbar(np.mean(vals), i, xerr=s["std"], fmt="o", color=colr, ms=7, capsize=4)
                ax.scatter(vals, [i + 0.2] * len(vals), color=colr, s=12, alpha=0.6)
                txt = f"{np.mean(vals):.3g} ± {s['std']:.2g} ({len(vals)} seeds)"
            ax.text(0.98, i - 0.22, txt, transform=ax.get_yaxis_transform(), ha="right", fontsize=8)
        ax.set(yticks=range(len(rows)), yticklabels=[f"{r[0]} / {r[2]}" for r in rows],
               ylim=(len(rows) - 0.5, -0.6))
        ax.set_title(title, loc="left")
    dd.axvline(seeds["sequence"]["map"]["OOD AUROC"]["mean"], c="k", ls="--", lw=1, label="MAP entropy")
    dd.axvline(0.5, c="0.6", lw=0.8)
    dd.set(xlim=(-0.05, 1.05), xlabel="AUROC")
    dd.legend(fontsize=8, loc="lower center")
    e.set(xlim=(0.6, 0.8), xlabel="AUROC")
    f.axvline(0, c="k", ls=":", lw=1)
    f.set(xscale="symlog", xlabel="nats (symlog; MAP = 0)")
    f.set_xscale("symlog", linthresh=1e-4)
    f.set_xlim(-1e-4, 0.05)

    fig.suptitle("Constraints: sequence mode (256 / 1792 sequences) vs token mode (256), measured α, tol 1%",
                 fontsize=14)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
