"""Posterior samples that remove only the top-k directions of J^L instead of
projecting onto its whole kernel (llmproj.lowrank), for several k at once.

One randomized subspace iteration finds k_max directions; every smaller k
uses the leading k of them. For each k the samples are

    delta_k = (I - U_k U_k^T) eps / sqrt(alpha_k)

with the same eps as sample_posterior.py at the same --seed (so the exact
token-mode posterior is the k -> rank limit on identical noise) and alpha_k
measured like the exact posterior's (worst per-sequence |dNLL| reaches
--alpha-tol of the MAP's mean NLL). Saved as methods topk<k>_measured in a
token-mode samples file that evaluate_posterior.py / analyze_posterior.py read.

Also reports J^L's spectrum: the Ritz values and the share of ||J^L||_F^2
in the top k directions, and how far each delta_k still is from the kernel
(||J^L delta|| / ||delta||, the exact run's residual being ~0.02).

  uv run python scripts/sample_topk_posterior.py --ks 16 64 256 1024 2048
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.alpha import measure_alpha, per_row_losses
from llmproj.data import encode_texts, load_texts
from llmproj.losses import make_causal_lm_loss
from llmproj.lowrank import project_out, top_singular_subspace
from llmproj.paramspace import ParamSpace
from llmproj.projection import kernel_residuals


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--split", default="train[:2000]")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--proj-batch-size", type=int, default=8)
    ap.add_argument("--loss-on", choices=["all", "response"], default=None,
                    help="default: as the adapter was fine-tuned")
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 64, 256, 1024, 2048])
    ap.add_argument("--n-iter", type=int, default=3, help="subspace iterations")
    ap.add_argument("--vec-batch", type=int, default=16, help="vectors per vmapped pass")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--alpha-tol", type=float, default=0.01)
    ap.add_argument("--out", default="samples/posterior_topk.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()
    run = Path(args.adapter) / "run.json"
    if args.loss_on is None:
        args.loss_on = json.loads(run.read_text()).get("loss_on", "all") if run.exists() else "all"

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, attn_implementation="eager",
                                                dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta, frozen = space.theta(model), space.frozen(model)
    f = make_causal_lm_loss(model, space, frozen, mode="token")
    f_seq = make_causal_lm_loss(model, space, frozen, mode="sequence")

    texts = load_texts(args.dataset, args.split, args.text_field)[: args.n_train]
    enc = encode_texts(tokenizer, texts, args.max_length, loss_on=args.loss_on)
    S = args.proj_batch_size
    batches = []
    for i in range(0, enc["input_ids"].size(0), S):  # as in sample_posterior.py
        L = int(enc["attention_mask"][i : i + S].sum(1).max())
        batches.append({k: v[i : i + S, :L].to(args.device) for k, v in enc.items()})
    print(f"{space}; {len(batches)} batches, {int(enc['loss_mask'][:, 1:].sum())} target tokens")

    ks = sorted(set(args.ks))
    sub = top_singular_subspace(f, theta, space, batches, k=ks[-1], n_iter=args.n_iter,
                                vec_batch=args.vec_batch, seed=args.seed + 7)
    U, sigma2, frob2 = sub["U"], sub["sigma2"], sub["frob2_est"]
    t_sub = time.perf_counter() - t0
    print(f"subspace of {ks[-1]} directions in {t_sub / 3600:.2f} h; "
          f"||J||_F^2 ~ {frob2:.4g}")

    # the same eps as sample_projected_posterior at this seed
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    eps = torch.stack([torch.randn(space.P, generator=gen, dtype=torch.float32)
                       for _ in range(args.n_samples)]).to(space.device, space.dtype)

    map_rows = per_row_losses(f_seq, theta, batches)
    tol_abs = args.alpha_tol * float(map_rows.mean())
    methods, per_k = {}, {}
    for k in ks:
        dirs = project_out(eps, U[:k])
        kdim = float(space.P - k)
        alpha, radii = measure_alpha(f_seq, theta, space, batches, dirs, kdim, tol_abs)
        res = kernel_residuals(dirs, f, theta, space, batches)
        methods[f"topk{k}_measured"] = (dirs / alpha ** 0.5).float().cpu()
        per_k[k] = {"alpha": alpha, "norm": (kdim / alpha) ** 0.5,
                    "energy_share": float(sigma2[:k].sum()) / frob2,
                    "residual_mean": float(res.mean()), "residual_max": float(res.max()),
                    "radii": radii}
        print(f"  k={k:5d}: energy share {per_k[k]['energy_share']:.3f}  residual "
              f"{per_k[k]['residual_mean']:.3f} (max {per_k[k]['residual_max']:.3f})  "
              f"alpha {alpha:.4g}  ||delta|| ~ {per_k[k]['norm']:.3f}")

    info = {"P": space.P, "loss_mode": "token", "loss_on": args.loss_on, "projector": "topk",
            "alpha_tol_nats": tol_abs, "ks": ks, "per_k": per_k, "sigma2": sigma2.tolist(),
            "frob2_est": frob2, "kernel_dim": float(space.P - ks[-1]),
            "timings": {"subspace": t_sub, "total": time.perf_counter() - t0}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"methods": methods, "names": space.names, "shapes": space.shapes,
                "info": info, "args": vars(args)}, out)
    out.with_suffix(".json").write_text(json.dumps({"info": info, "args": vars(args)}, indent=2))
    print(f"saved {out} ({(time.perf_counter() - t0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
