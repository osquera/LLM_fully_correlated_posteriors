"""Sweep the loss tolerance behind the measured alpha, without re-projecting.

A projected sample is (I - P) eps / sqrt(alpha): the projection does not
depend on alpha. So the vectors saved by sample_posterior.py, multiplied back
by sqrt(alpha_measured), are the alpha = 1 projections, and any other alpha is
a rescaling. That matters for token mode, whose projection took ~22 GPU hours.

For each --tols value (a fraction of the MAP's mean NLL on the projection
rows, as sample_posterior.py's --alpha-tol):

  1. measure the radius r* per saved direction under the per-sequence
     criterion (one grid scan per direction serves every tolerance), and set
     alpha = kernel_dim / min(r*)^2, separately per loss mode;
  2. rescale the projected and isotropic samples to that alpha;
  3. for the sequence file, redraw diagonal Laplace and LLA at the sequence
     alpha from a rebuilt dense J^L (they are not a rescaling: the row-space
     variance is 1 / (s_i^2 + alpha));
  4. evaluate everything as evaluate_posterior.py does.

Writes figures/tolerance_sweep.json; plot with scripts/plot_tolerance_sweep.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_posterior import build_sets, evaluate_method, print_selective
from llmproj.alpha import loss_radii, per_row_losses
from llmproj.baselines import (
    build_loss_jacobian,
    diagonal_laplace_samples,
    linearized_laplace_samples,
)
from llmproj.data import encode_texts, load_texts
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+",
                    default=["samples/posterior_sequence.pt", "samples/posterior_token.pt"],
                    help="files from sample_posterior.py; missing ones are skipped")
    ap.add_argument("--tols", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--test-split", default="train[2000:2256]")
    ap.add_argument("--ood-dataset", default="wikitext")
    ap.add_argument("--ood-config", default="wikitext-2-raw-v1")
    ap.add_argument("--ood-split", default="test")
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-bisect", type=int, default=6)
    ap.add_argument("--out", default="figures/tolerance_sweep.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    files = [Path(p) for p in args.samples if Path(p).exists()]
    if not files:
        raise SystemExit(f"none of {args.samples} exist")
    blobs = {}
    for p in files:
        blob = torch.load(p, weights_only=False)
        blobs[blob["info"]["loss_mode"]] = blob
    sargs = next(iter(blobs.values()))["args"]
    tols = sorted(args.tols)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta = space.theta(model)
    frozen = space.frozen(model)
    f_seq = make_causal_lm_loss(model, space, frozen, mode="sequence")

    # the projection rows, batched as the sequence sampler did
    texts = load_texts(sargs["dataset"], sargs["split"], sargs["text_field"])
    enc = encode_texts(tokenizer, texts[: sargs["n_train"]], sargs["max_length"],
                       loss_on=sargs.get("loss_on", "all"))
    S = 16
    batches = [{k: v[i : i + S].to(args.device) for k, v in enc.items()}
               for i in range(0, enc["input_ids"].size(0), S)]
    map_rows = per_row_losses(f_seq, theta, batches)
    tols_nats = [t * float(map_rows.mean()) for t in tols]
    grid = torch.logspace(-1, 3, 41)

    # alpha = 1 projections and isotropic draws, recovered from the saved samples
    unit, alphas, radii = {}, {}, {}
    for mode, blob in blobs.items():
        a0 = blob["info"]["alphas"]["measured"]
        pv = blob["methods"]["projected_measured"] * a0 ** 0.5
        iso = blob["methods"]["isotropic_measured"] * a0 ** 0.5
        kdim = blob["info"]["kernel_dim"]
        unit[mode] = {"projected": pv, "isotropic": iso}
        t0 = time.perf_counter()
        per_dir = [loss_radii(f_seq, theta, space, batches, d, tols_nats, base=map_rows,
                              grid=grid, n_bisect=args.n_bisect) for d in pv]
        radii[mode] = [[r[i] for r in per_dir] for i in range(len(tols))]  # [tol][dir]
        alphas[mode] = [kdim / min(r) ** 2 for r in radii[mode]]
        print(f"{mode}: radii for {len(tols)} tolerances x {len(pv)} directions "
              f"in {time.perf_counter() - t0:.0f}s")
        for t, r, a in zip(tols, radii[mode], alphas[mode]):
            print(f"  tol {t:<6g} r* = {min(r):8.4g}  (median {sorted(r)[len(r) // 2]:.4g})  "
                  f"alpha = {a:.4e}")

    J = None
    if "sequence" in blobs:
        J = build_loss_jacobian(f_seq, theta, space, batches, max_gb=8.0)
    seed = sargs["seed"]
    n_seq_samples = unit["sequence"]["projected"].shape[0] if "sequence" in blobs else 0

    sets = build_sets(tokenizer, sargs, args)
    map_res, map_seq_nll = evaluate_method(model, frozen, theta, space, torch.zeros(1, space.P),
                                           sets, args.device, args.batch_size)
    map_seq_nll = map_seq_nll[0]

    sweep = []
    for i, tol in enumerate(tols):
        methods = {}
        for mode in blobs:
            a = alphas[mode][i]
            for m, v in unit[mode].items():
                methods[f"{mode}/{m}"] = v / a ** 0.5
        if J is not None:
            a = alphas["sequence"][i]
            methods["sequence/diagonal_laplace"] = diagonal_laplace_samples(
                J, a, n_seq_samples, seed=seed + 2)
            methods["sequence/lla"] = linearized_laplace_samples(
                J, a, n_seq_samples, seed=seed + 3, rtol=sargs["rtol"])
        results = {"map": map_res}
        for name, deltas in methods.items():
            results[name], _ = evaluate_method(model, frozen, theta, space, deltas.float(), sets,
                                               args.device, args.batch_size, map_seq_nll)
        print(f"\n{'=' * 30} tol {tol:g} ({tols_nats[i]:.4g} nats) {'=' * 30}")
        print(f"{'method':32s} {'norm':>7s} {'trainΔ':>8s} {'test NLL':>9s} {'ECE':>7s} "
              f"{'test MI':>9s} {'ood MI':>9s} {'AUROC':>6s}")
        for name, r in results.items():
            print(f"{name:32s} {r['norm']:7.3g} {r['train']['worst_seq_dnll']:8.3g} "
                  f"{r['test']['nll_bma']:9.4f} {r['test']['ece']:7.4f} {r['test']['mi']:9.3g} "
                  f"{r['ood']['mi']:9.3g} {r['ood']['auroc']:6.3f}")
        print_selective(results)
        sweep.append({"tol": tol, "tol_nats": tols_nats[i],
                      "alphas": {m: alphas[m][i] for m in blobs},
                      "radii": {m: radii[m][i] for m in blobs},
                      "results": results})

        out = Path(args.out)  # rewritten after every tolerance, in case the job is cut short
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"sweep": sweep, "args": vars(args),
                                   "sampler_args": {m: b["args"] for m, b in blobs.items()}},
                                  indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
