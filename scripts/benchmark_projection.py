"""Time the token-mode projection, one-at-a-time vs batched, on the real model.

Measures, on the first --n-batches batches of the projection data:

  precompute  forming one batch's J J^T, for each --row-batch
              (1 = rows one at a time, as before batching)
  sweep       one sweep of --n-vectors vectors over those batches, for each
              --vec-batch (1 = vectors one at a time, as before batching)

and extrapolates each to a full run (all batches; --n-iterations sweeps for
--n-samples samples), with the peak GPU memory. Out-of-memory settings are
reported and skipped.

  uv run python scripts/benchmark_projection.py --adapter checkpoints/smollm2_lora
"""

from __future__ import annotations

import argparse
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.data import encode_texts, load_texts
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace
from llmproj.projection import _jjt, precompute_batch_pinv, project_vectors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--split", default="train[:2000]")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--loss-on", default="all", choices=["all", "response"])
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=256, help="of the full run, to extrapolate")
    ap.add_argument("--proj-batch-size", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=2, help="batches actually timed")
    ap.add_argument("--n-samples", type=int, default=8, help="of the full run")
    ap.add_argument("--n-iterations", type=int, default=500, help="of the full run")
    ap.add_argument("--row-batch", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--vec-batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--repeats", type=int, default=2, help="timed sweeps per setting")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device):
    """(seconds, peak GB) of one call, or (None, None) on out-of-memory."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    t0 = time.perf_counter()
    try:
        fn()
    except torch.OutOfMemoryError:
        return None, None
    sync(device)
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else float("nan")
    return time.perf_counter() - t0, peak


def main():
    args = parse_args()
    if args.device.startswith("cuda"):
        print(torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, attn_implementation="eager",
                                                dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta, frozen = space.theta(model), space.frozen(model)
    f = make_causal_lm_loss(model, space, frozen, mode="token")

    texts = load_texts(args.dataset, args.split, args.text_field)[: args.n_train]
    enc = encode_texts(tokenizer, texts, args.max_length, loss_on=args.loss_on)
    S = args.proj_batch_size
    batches = []
    for i in range(0, enc["input_ids"].size(0), S):  # as in sample_posterior.py
        L = int(enc["attention_mask"][i : i + S].sum(1).max())
        b = {k: v[i : i + S, :L].to(args.device) for k, v in enc.items()}
        b["row_mask"] = b["loss_mask"][:, 1:].reshape(-1).bool()
        batches.append(b)
    n_full = len(batches)
    timed_batches = batches[: args.n_batches]
    rows = [int(b["row_mask"].sum()) for b in timed_batches]
    print(f"{space}, {n_full} batches in the full run; timing {len(timed_batches)} "
          f"with {sum(rows)} non-zero rows")

    print("\nprecompute (J J^T of one batch):")
    b0 = timed_batches[0]
    fb = lambda p: f(p, b0)  # noqa: E731
    R = int(fb(theta).numel())
    idx = b0["row_mask"].nonzero().squeeze(1).cpu()
    base_t = None
    for rb in args.row_batch:
        t, peak = timed(lambda: _jjt(fb, theta, space, R, idx, row_batch=rb), args.device)
        if t is None:
            print(f"  row_batch {rb:4d}: out of memory")
            continue
        base_t = base_t or t
        full_h = t / rows[0] * sum(int(b["row_mask"].sum()) for b in batches) / 3600
        print(f"  row_batch {rb:4d}: {t:7.2f} s  ({rows[0]} rows)  speed-up {base_t / t:5.2f}x  "
              f"peak {peak:5.1f} GB  -> full precompute ~{full_h:.2f} h")

    factors = precompute_batch_pinv(f, theta, space, timed_batches, progress=False)
    gen = torch.Generator().manual_seed(0)
    V = torch.randn(args.n_samples, space.P, generator=gen).to(space.device, space.dtype)

    print(f"\nsweep ({args.n_samples} vectors x {len(timed_batches)} batches, "
          f"{args.repeats} sweeps):")
    base_t = None
    for vb in args.vec_batch:
        def run():
            for i in range(0, args.n_samples, vb):
                project_vectors(V[i:i + vb], f, theta, space, timed_batches, factors,
                                n_iterations=args.repeats, tol=None)
        t, peak = timed(run, args.device)
        if t is None:
            print(f"  vec_batch {vb:3d}: out of memory")
            continue
        base_t = base_t or t
        per = t / args.repeats / len(timed_batches)  # all vectors, one batch, one sweep
        full_h = per * n_full * args.n_iterations / 3600
        print(f"  vec_batch {vb:3d}: {t:7.2f} s  speed-up {base_t / t:5.2f}x  peak {peak:5.1f} GB"
              f"  -> full projection ({args.n_iterations} sweeps) ~{full_h:.2f} h")


if __name__ == "__main__":
    main()
