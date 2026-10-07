"""Stage 2: draw posterior samples over the LoRA weights, for several methods.

The projected posterior is q(theta) = N(theta_map, alpha^-1 U_L U_L^T), with
U_L spanning ker(J^L). A sample is (I - P) eps / sqrt(alpha), so we project the
eps once at alpha = 1 and rescale afterwards: every alpha below reuses the same
projected vectors.

Two choices of alpha are saved side by side:

  measured     the largest prior scale at which every projection row keeps its
               mean NLL within --alpha-tol of theta_map (README: "Ways out").
               The criterion is always the per-sequence mean NLL, whatever
               --loss-mode is, so the two loss modes are compared on the same
               footing.
  closed_form  Lemma 3.4 (corrected), alpha* = rank(J^L) / ||theta_map||^2.

Methods saved (each an (n_samples, P) tensor of theta - theta_map):

  projected_{measured,closed_form}   the projected posterior
  isotropic_{measured,closed_form}   N(0, alpha^-1 I), the unprojected control
  diagonal_laplace_measured          (diag(J^T J) + alpha I)^-1   [exact path only]
  lla_measured                       (J^T J + alpha I)^-1         [exact path only]

The Laplace baselines share the measured alpha, which puts all methods at very
nearly the same sample norm (P - rank is close to P), i.e. a matched-norm
comparison.

Projector: "exact" builds the dense J^L and takes its SVD (sequence mode at
N = 256 needs ~0.5 GB); "iterative" uses alternating projections (token mode,
where dense J^L would need ~120 GB). "auto" picks exact when it fits --max-gb.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.alpha import measure_alpha, optimal_alpha, per_row_losses
from llmproj.baselines import (
    build_loss_jacobian,
    diagonal_laplace_samples,
    exact_kernel_projections,
    jacobian_memory_gb,
    linearized_laplace_samples,
)
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace
from llmproj.projection import kernel_residual, sample_projected_posterior


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--split", default="train[:2000]",
                    help="MUST be the fine-tuning split: we project against theta_map's own data")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=256, help="sequences behind J^L")
    ap.add_argument("--proj-batch-size", type=int, default=16, help="S; per-batch inverse is S x S")
    ap.add_argument("--loss-mode", choices=["sequence", "token"], default="sequence")
    ap.add_argument("--projector", choices=["auto", "exact", "iterative"], default="auto")
    ap.add_argument("--max-gb", type=float, default=8.0, help="dense J^L budget for --projector auto")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--n-iterations", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-3, help="early-stop kernel residual")
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative eigenvalue cutoff")
    ap.add_argument("--no-acceleration", action="store_true")
    ap.add_argument("--alpha-tol", type=float, default=0.01,
                    help="measured alpha: max |change in a sequence's mean NLL|, as a "
                         "fraction of the MAP's mean NLL over the projection rows")
    ap.add_argument("--alpha-directions", type=int, default=None,
                    help="samples whose directions are probed (default: all of them, "
                         "so every saved sample is within tolerance)")
    ap.add_argument("--out", default=None, help="default: samples/posterior_<loss-mode>.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out or f"samples/posterior_{args.loss_mode}.pt")
    timings = {}

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device)
    model.eval()

    # theta = the LoRA adapters only. Everything else is frozen.
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta = space.theta(model)
    frozen = space.frozen(model)
    print(space)

    f = make_causal_lm_loss(model, space, frozen, mode=args.loss_mode)
    f_seq = make_causal_lm_loss(model, space, frozen, mode="sequence")

    ds = load_dataset(args.dataset, split=args.split)
    field = args.text_field if args.text_field in ds.column_names else ds.column_names[0]
    ds = ds.select(range(min(args.n_train, len(ds))))
    enc = tokenizer(
        list(ds[field]), truncation=True, max_length=args.max_length,
        padding="max_length", return_tensors="pt",
    )
    S = args.proj_batch_size
    batches = [
        {"input_ids": enc["input_ids"][i : i + S].to(args.device),
         "attention_mask": enc["attention_mask"][i : i + S].to(args.device)}
        for i in range(0, enc["input_ids"].size(0), S)
    ]
    n_rows = int(sum(f(theta, b).numel() for b in batches))
    print(f"{len(batches)} batches, {n_rows} J^L rows (mode={args.loss_mode})")

    projector = args.projector
    if projector == "auto":
        need = jacobian_memory_gb(n_rows, space.P, space.dtype)
        projector = "exact" if need <= args.max_gb else "iterative"
        print(f"dense J^L would take {need:.2f} GB -> projector = {projector}")

    # ---- project once, at alpha = 1 ----
    t0 = time.perf_counter()
    J = None
    if projector == "exact":
        J = build_loss_jacobian(f, theta, space, batches, max_gb=args.max_gb)
        pv, rank = exact_kernel_projections(J, args.n_samples, seed=args.seed, rtol=args.rtol)
        kdim, n_probes = float(space.P - rank), None
        info = {"rank": rank, "kernel_residual_max": max(
            kernel_residual(v.to(space.device, space.dtype), f, theta, space, batches)
            for v in pv)}
    else:
        pv, info = sample_projected_posterior(
            f, theta, space, batches,
            n_samples=args.n_samples, alpha=1.0,
            n_iterations=args.n_iterations, rtol=args.rtol,
            acceleration=not args.no_acceleration, tol=args.tol, seed=args.seed,
        )
        # Hutchinson over the samples themselves; std sqrt(2 kdim / n_samples)
        kdim, n_probes = float(info["kernel_dim_est"]), args.n_samples
        if info["kernel_residual_max"] > 10 * args.tol:
            print(f"WARNING: max kernel residual {info['kernel_residual_max']:.3e} is far "
                  f"above tol={args.tol}. Lemma 4.3 does not hold at this residual -- "
                  f"raise --n-iterations.")
    timings["projection"] = time.perf_counter() - t0
    print(f"projected {pv.shape[0]} vectors in {timings['projection']:.1f}s, "
          f"kernel_dim = {kdim:.1f} / P = {space.P}, "
          f"max residual = {info['kernel_residual_max']:.2e}")

    # ---- alpha: closed form and measured ----
    alpha_cf = optimal_alpha(theta, space, kdim, n_probes=n_probes)

    t0 = time.perf_counter()
    map_rows = per_row_losses(f_seq, theta, batches)
    tol_abs = args.alpha_tol * float(map_rows.mean())
    k = args.alpha_directions or pv.shape[0]
    alpha_ms, radii = measure_alpha(f_seq, theta, space, batches, pv[:k], kdim, tol_abs)
    timings["measure_alpha"] = time.perf_counter() - t0

    alphas = {"measured": alpha_ms, "closed_form": alpha_cf}
    for name, a in alphas.items():
        print(f"alpha[{name:11s}] = {a:.4e}   => ||delta|| ~ {(kdim / a) ** 0.5:.4g}")
    print(f"  measured: tol = {tol_abs:.4g} nats ({args.alpha_tol:g} x MAP mean NLL "
          f"{float(map_rows.mean()):.4f}); radii min/median/max = "
          f"{min(radii):.4g} / {sorted(radii)[len(radii) // 2]:.4g} / {max(radii):.4g}")

    # ---- methods ----
    gen = torch.Generator().manual_seed(args.seed + 1)
    iso = torch.randn(args.n_samples, space.P, generator=gen)
    methods = {}
    for name, a in alphas.items():
        methods[f"projected_{name}"] = pv / a**0.5
        methods[f"isotropic_{name}"] = iso / a**0.5
    if J is not None:
        a = alphas["measured"]
        methods["diagonal_laplace_measured"] = diagonal_laplace_samples(
            J, a, args.n_samples, seed=args.seed + 2)
        methods["lla_measured"] = linearized_laplace_samples(
            J, a, args.n_samples, seed=args.seed + 3, rtol=args.rtol)

    norms = {m: float(d.norm(dim=1).mean()) for m, d in methods.items()}
    print("\nmean sample norm per method:")
    for m, n in norms.items():
        print(f"  {m:28s} {n:10.4g}")

    info.update({
        "P": space.P, "n_rows": n_rows, "loss_mode": args.loss_mode,
        "projector": projector, "kernel_dim": kdim, "alphas": alphas,
        "alpha_tol_nats": tol_abs, "radii": radii, "norms": norms,
        "theta_norm_sq": float(sum((t.float() ** 2).sum() for t in theta.values())),
        "timings": timings,
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"methods": {m: d.float() for m, d in methods.items()},
                "names": space.names, "shapes": space.shapes,
                "info": info, "args": vars(args)}, out)
    out.with_suffix(".json").write_text(json.dumps(
        {"info": {k_: v for k_, v in info.items() if k_ != "batch_ranks"},
         "args": vars(args)}, indent=2))
    print(f"\nsaved {len(methods)} methods x {args.n_samples} samples to {out}")


if __name__ == "__main__":
    main()
