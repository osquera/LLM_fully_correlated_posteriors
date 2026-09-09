"""Stage 2: draw samples from the loss-projected posterior over LoRA weights.

q(theta) = N(theta_map, alpha^-1 U_L U_L^T), U_L spanning ker(J^L).
alpha comes from Lemma 3.4 in closed form unless --alpha is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.alpha import estimate_kernel_dim, optimal_alpha
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace
from llmproj.projection import precompute_batch_pinv, sample_projected_posterior


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--split", default="train[:2000]",
                    help="MUST be the fine-tuning split: we project against theta_map's own data")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=256, help="N rows of J^L")
    ap.add_argument("--proj-batch-size", type=int, default=16, help="S; per-batch inverse is S x S")
    ap.add_argument("--loss-mode", choices=["sequence", "token"], default="sequence")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--n-iterations", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-3, help="early-stop kernel residual")
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative eigenvalue cutoff")
    ap.add_argument("--alpha", type=float, default=None, help="default: Lemma 3.4 closed form")
    ap.add_argument("--no-acceleration", action="store_true")
    ap.add_argument("--out", default="samples/projected.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device)
    model.eval()

    # theta = the LoRA adapters only. Everything else is frozen.
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"])
    theta = space.theta(model)
    frozen = space.frozen(model)
    print(space)

    f = make_causal_lm_loss(model, space, frozen, mode=args.loss_mode)

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
    rows_per_batch = S if args.loss_mode == "sequence" else S * (args.max_length - 1)
    print(f"{len(batches)} batches, {rows_per_batch} J^L rows each "
          f"=> per-batch inverse is {rows_per_batch} x {rows_per_batch}")

    factors = precompute_batch_pinv(f, theta, space, batches, rtol=args.rtol)

    if args.alpha is None:
        kdim = estimate_kernel_dim(
            f, theta, space, batches, factors,
            n_probes=8, n_iterations=args.n_iterations,
            acceleration=not args.no_acceleration, seed=args.seed,
        )
        alpha = optimal_alpha(theta, space, kdim)
        print(f"Lemma 3.4: kernel_dim ~ {kdim:.1f} / P = {space.P}  =>  alpha* = {alpha:.6e}")
    else:
        alpha, kdim = args.alpha, float("nan")
        print(f"using supplied alpha = {alpha:.6e}")

    deltas, info = sample_projected_posterior(
        f, theta, space, batches,
        n_samples=args.n_samples, alpha=alpha,
        n_iterations=args.n_iterations, rtol=args.rtol,
        acceleration=not args.no_acceleration, tol=args.tol,
        seed=args.seed, factors=factors,
    )
    info["kernel_dim_hutchinson"] = kdim
    info["loss_mode"] = args.loss_mode
    print(json.dumps({k: v for k, v in info.items() if k != "batch_ranks"}, indent=2))
    if info["kernel_residual_max"] > 10 * args.tol:
        print(f"WARNING: max kernel residual {info['kernel_residual_max']:.3e} is far above "
              f"tol={args.tol}. Lemma 4.3 does not hold at this residual -- raise --n-iterations.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"deltas": deltas, "names": space.names, "shapes": space.shapes,
         "info": info, "args": vars(args)}, out,
    )
    print(f"saved {deltas.shape[0]} samples ({deltas.shape[1]:,} params each) to {out}")


if __name__ == "__main__":
    main()
