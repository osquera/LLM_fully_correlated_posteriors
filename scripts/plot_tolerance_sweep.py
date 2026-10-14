"""Plot figures/tolerance_sweep.json (from sweep_tolerance.py).

Six panels. The top row puts every method on its sample norm ||delta||, the
matched-scale comparison; the bottom row is about uncertainty, per tolerance.

  a  Lemma 4.3: worst change in a train sequence's NLL
  b  held-out BMA NLL, relative to the MAP
  c  held-out ECE
  d  OOD detection AUROC from MI (dashed: the MAP's entropy)
  e  is the model wrong where uncertain? error AUROC, pooled test + ood, from
     MI (solid) and from the BMA's 1 - confidence (dotted); the MAP's own
     scores as horizontal references
  f  accuracy on the most certain half of the tokens (by BMA entropy)

Colour = method, line style = loss mode (solid sequence, dashed token).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter

COLOR = {"projected": "C2", "isotropic": "C7", "diagonal_laplace": "C1", "lla": "C4"}
LS = {"sequence": "-", "token": "--"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="figures/tolerance_sweep.json")
    ap.add_argument("--out", default="figures/tolerance_sweep.png")
    args = ap.parse_args()
    sweep = json.loads(Path(args.sweep).read_text())["sweep"]
    tols = [100 * s["tol"] for s in sweep]
    methods = [m for m in sweep[0]["results"] if m != "map"]
    mp = sweep[0]["results"]["map"]

    def series(m, get):
        return [get(s["results"][m]) for s in sweep]

    def style(m):
        mode, base = m.split("/")
        return dict(color=COLOR[base], ls=LS[mode], marker="o" if base == "projected" else ".",
                    ms=5, lw=2.2 if base == "projected" else 1.3,
                    label=f"{mode} / {base.replace('_', ' ')}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    (a, b, c), (d, e, f) = axes

    for m in methods:
        norm = series(m, lambda r: r["norm"])
        a.plot(norm, series(m, lambda r: r["train"]["worst_seq_dnll"]), **style(m))
        b.plot(norm, series(m, lambda r: r["test"]["nll_bma"] - mp["test"]["nll_bma"]), **style(m))
        c.plot(norm, series(m, lambda r: r["test"]["ece"]), **style(m))
        d.plot(tols, series(m, lambda r: r["ood"]["auroc"]), **style(m))
        f.plot(tols, series(m, lambda r: r["pooled"]["selective"]["entropy"]["acc@0.5"]), **style(m))
        if m.endswith("projected"):
            e.plot(tols, series(m, lambda r: r["pooled"]["selective"]["mi"]["err_auroc"]), **style(m))
            st = style(m) | {"ls": ":", "marker": None, "lw": 1.5,
                             "label": f"{m.split('/')[0]} / projected, BMA 1-conf"}
            e.plot(tols, series(m, lambda r: r["pooled"]["selective"]["1-conf"]["err_auroc"]), **st)
        else:
            e.plot(tols, series(m, lambda r: r["pooled"]["selective"]["mi"]["err_auroc"]),
                   **style(m) | {"label": None})

    a.set(xscale="log", yscale="log", xlabel="mean sample norm ‖δ‖",
          ylabel="worst |Δ train-sequence NLL| (nats)")
    a.set_title("a  Lemma 4.3: train loss under samples", loc="left")

    b.axhline(0, c="k", ls=":", lw=1)
    b.set(xscale="log", yscale="symlog", xlabel="mean sample norm ‖δ‖",
          ylabel="test BMA NLL − MAP (nats)")
    b.set_yscale("symlog", linthresh=1e-3)
    b.set_title(f"b  held-out NLL (MAP {mp['test']['nll_bma']:.4f}; above 0 = worse)", loc="left")

    c.axhline(mp["test"]["ece"], c="k", ls=":", lw=1, label="MAP")
    c.set(xscale="log", xlabel="mean sample norm ‖δ‖", ylabel="test ECE")
    c.set_title("c  held-out calibration (lower better)", loc="left")
    c.legend(handles=c.get_lines()[-1:], fontsize=8, loc="upper left")

    d.axhline(mp["ood"]["auroc"], c="k", ls=":", lw=1, label="MAP entropy")
    d.axhline(0.5, c="0.6", lw=0.8)
    d.set(xscale="log", xlabel="tolerance (% of MAP mean train NLL)",
          ylabel="AUROC, test vs WikiText, by MI", ylim=(-0.03, 1.03))
    d.set_title("d  OOD detection from epistemic uncertainty", loc="left")
    d.legend(fontsize=8, loc="center left", handles=d.get_lines()[-2:-1])

    ms = mp["pooled"]["selective"]
    e.axhline(ms["1-conf"]["err_auroc"], c="k", ls=":", lw=1.2, label="MAP 1-conf")
    e.axhline(ms["entropy"]["err_auroc"], c="k", ls="-.", lw=1, label="MAP entropy")
    e.set(xscale="log", xlabel="tolerance (% of MAP mean train NLL)",
          ylabel="error AUROC (wrong vs right token)")
    e.set_title("e  wrong where uncertain? solid/dashed = MI", loc="left")
    refs = [l for l in e.get_lines() if l.get_label().startswith(("MAP", "sequence / projected, BMA",
                                                                 "token / projected, BMA"))]
    e.legend(handles=refs, fontsize=8, loc="center right")

    f.axhline(ms["entropy"]["acc@0.5"], c="k", ls=":", lw=1, label="MAP")
    f.set(xscale="log", xlabel="tolerance (% of MAP mean train NLL)",
          ylabel="accuracy on the 50% most certain tokens")
    f.set_title("f  selective prediction, test + OOD pooled", loc="left")
    f.legend(handles=f.get_lines()[-1:], fontsize=8, loc="lower left")

    for ax in axes.flat:
        ax.grid(alpha=0.3, which="both", lw=0.5)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax_ticks = [3, 5, 10, 20, 30]
    for ax in (a, b, c):
        ax.set_xticks(ax_ticks)
    for ax in (d, e, f):
        ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    handles = [l for l in a.get_lines()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=10,
               frameon=False, handlelength=3.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
