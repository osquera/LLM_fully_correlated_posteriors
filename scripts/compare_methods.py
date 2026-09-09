"""Compare projected / linearised-Laplace / diagonal-Laplace posteriors.

Reports, for each method: wall-clock cost, how much each sample perturbs the
training loss (Lemma 4.3 / the underfitting claim), and how far each sample is
from ker(J^L). Runs on a tiny random Llama by default so it needs no download;
point --adapter at a real checkpoint to run it for real.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from llmproj.baselines import (build_loss_jacobian, diagonal_laplace_samples,
                               exact_projection_samples, jacobian_memory_gb,
                               linearized_laplace_samples)
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace
from llmproj.projection import (kernel_residual, precompute_batch_pinv,
                                sample_projected_posterior)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint; omit for a tiny random model")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--proj-batch-size", type=int, default=4)
    ap.add_argument("--loss-mode", choices=["sequence", "token"], default="sequence")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--n-iterations", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--max-gb", type=float, default=8.0)
    ap.add_argument("--out", default="figures/compare_methods.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def tiny_model(seq_len, device):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=seq_len, attn_implementation="eager")
    m = LlamaForCausalLM(cfg).float()
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    return m.to(device).eval(), 256


def real_model(args):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", dtype=torch.float32)
    m = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    return m, tok.vocab_size


def main():
    args = parse_args()
    torch.manual_seed(0)

    if args.adapter is None:
        model, vocab = tiny_model(args.seq_len, args.device)
        print("using a tiny randomly-initialised Llama (no download)")
    else:
        model, vocab = real_model(args)

    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"])
    theta, frozen = space.theta(model), space.frozen(model)
    f = make_causal_lm_loss(model, space, frozen, mode=args.loss_mode)

    ids = torch.randint(0, vocab, (args.n_seq, args.seq_len), device=args.device)
    S = args.proj_batch_size
    batches = [
        {"input_ids": ids[i : i + S],
         "attention_mask": torch.ones(min(S, args.n_seq - i), args.seq_len,
                                      dtype=torch.long, device=args.device)}
        for i in range(0, args.n_seq, S)
    ]
    n_rows = int(sum(f(theta, b).numel() for b in batches))
    print(f"{space}  |  {len(batches)} batches, {n_rows} J^L rows total, "
          f"mode={args.loss_mode}")
    print(f"dense J^L would take {jacobian_memory_gb(n_rows, space.P):.3f} GB")

    def dloss(delta):
        pert = {k: theta[k] + d for k, d in space.unflatten(delta.to(space.dtype)).items()}
        return max(float((f(pert, b) - f(theta, b)).abs().max()) for b in batches)

    def resid(delta):
        return kernel_residual(delta.to(space.dtype).to(space.device), f, theta, space, batches)

    results, timings = {}, {}

    # ---- shared dense J^L: baselines and the exact projector all need it ----
    t0 = time.perf_counter()
    J = build_loss_jacobian(f, theta, space, batches, max_gb=args.max_gb, progress=False)
    timings["build_dense_J"] = time.perf_counter() - t0
    print(f"built dense J^L {tuple(J.shape)} in {timings['build_dense_J']:.2f}s")

    alpha = args.alpha
    if alpha is None:
        s = torch.linalg.svdvals(J.float())
        rank = int((s > 1e-6 * s.max()).sum())
        theta_norm_sq = float(sum((t.float() ** 2).sum() for t in theta.values()))
        alpha = theta_norm_sq / max(space.P - (space.P - rank), 1)
        print(f"Lemma 3.4: rank(J^L) = {rank}, alpha* = {alpha:.6e}")

    methods = {
        "projected_exact": lambda: exact_projection_samples(J, alpha, args.n_samples),
        "lla_full_cov": lambda: linearized_laplace_samples(J, alpha, args.n_samples),
        "diagonal_laplace": lambda: diagonal_laplace_samples(J, alpha, args.n_samples),
    }
    for name, fn in methods.items():
        t0 = time.perf_counter()
        d = fn()
        timings[name] = time.perf_counter() - t0
        results[name] = {
            "norm": float(d.norm(dim=1).mean()),
            "dloss": max(dloss(d[i]) for i in range(d.shape[0])),
            "residual": max(resid(d[i]) for i in range(d.shape[0])),
        }

    # ---- iterative projection (the scalable path) ----
    t0 = time.perf_counter()
    factors = precompute_batch_pinv(f, theta, space, batches, progress=False)
    timings["projected_iterative_precompute"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    d, info = sample_projected_posterior(
        f, theta, space, batches, n_samples=args.n_samples, alpha=alpha,
        n_iterations=args.n_iterations, tol=None, factors=factors, progress=False)
    timings["projected_iterative"] = time.perf_counter() - t0
    results["projected_iterative"] = {
        "norm": float(d.norm(dim=1).mean()),
        "dloss": max(dloss(d[i]) for i in range(d.shape[0])),
        "residual": info["kernel_residual_max"],
    }

    order = ["diagonal_laplace", "lla_full_cov", "projected_exact", "projected_iterative"]
    print(f"\n{'method':<30} {'time (s)':>10} {'||delta||':>11} {'max dloss':>12} {'residual':>11}")
    for k in order:
        r = results[k]
        t = timings[k] + (timings["projected_iterative_precompute"]
                          if k == "projected_iterative" else timings["build_dense_J"])
        print(f"{k:<30} {t:10.2f} {r['norm']:11.4f} {r['dloss']:12.4e} {r['residual']:11.3e}")
    print("\n(time includes each method's own preprocessing: dense J^L for the first")
    print(" three, per-batch (JJ^T)^+ for the iterative projection)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "timings": timings,
                               "alpha": alpha, "n_rows": n_rows, "P": space.P,
                               "args": vars(args)}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
