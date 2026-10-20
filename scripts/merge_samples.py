"""Merge posterior samples projected in separate jobs into one samples file.

A token-mode projection can outgrow one job's time limit (24 h on DTU HPC),
mostly in the per-sample sweeps. sample_posterior.py runs with a few samples
and a different --seed per job, in parallel; this script joins them as if one
job had drawn them all:

  * the alpha = 1 projections and isotropic draws are recovered from each
    part (the saved samples divided by that part's 1 / sqrt(alpha));
  * the kernel dimension is the Hutchinson mean over all parts' samples;
  * the measured alpha is measured again over *all* directions, so every
    merged sample is within the tolerance (each part only checked its own);
  * the closed-form alpha uses the pooled kernel dimension.

The output has the format of sample_posterior.py, so every later script
(evaluate, analyze, sweep, semantic) reads it unchanged.

  uv run python scripts/merge_samples.py samples/part_*.pt --out samples/posterior_token.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.alpha import measure_alpha, optimal_alpha, per_row_losses
from llmproj.data import encode_texts, load_texts
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace

# sample_posterior.py args that must agree between parts
SAME = ("base_model", "adapter", "dataset", "split", "text_field", "max_length", "n_train",
        "loss_mode", "loss_on", "projector", "n_iterations", "tol", "rtol", "alpha_tol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    blobs = [torch.load(p, weights_only=False) for p in args.parts]
    sargs = blobs[0]["args"]
    for p, b in zip(args.parts[1:], blobs[1:]):
        diff = {k: (sargs.get(k), b["args"].get(k)) for k in SAME if sargs.get(k) != b["args"].get(k)}
        if diff:
            raise SystemExit(f"{p} differs from {args.parts[0]}: {diff}")
    seeds = [b["args"]["seed"] for b in blobs]
    if len(set(seeds)) < len(seeds):
        raise SystemExit(f"parts share a seed ({seeds}): their samples would coincide")

    pv = torch.cat([b["methods"]["projected_measured"] * b["info"]["alphas"]["measured"] ** 0.5
                    for b in blobs])
    iso = torch.cat([b["methods"]["isotropic_measured"] * b["info"]["alphas"]["measured"] ** 0.5
                     for b in blobs])
    n = [b["methods"]["projected_measured"].shape[0] for b in blobs]
    kdim = sum(b["info"]["kernel_dim"] * k for b, k in zip(blobs, n)) / sum(n)
    print(f"{len(blobs)} parts (seeds {seeds}), {pv.shape[0]} samples, kernel_dim = {kdim:.1f}")

    tokenizer = AutoTokenizer.from_pretrained(sargs["adapter"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(sargs["base_model"], attn_implementation="eager",
                                                dtype=torch.float32)
    model = PeftModel.from_pretrained(base, sargs["adapter"]).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta, frozen = space.theta(model), space.frozen(model)
    f_seq = make_causal_lm_loss(model, space, frozen, mode="sequence")
    texts = load_texts(sargs["dataset"], sargs["split"], sargs["text_field"])[: sargs["n_train"]]
    enc = encode_texts(tokenizer, texts, sargs["max_length"], loss_on=sargs["loss_on"])
    S = sargs["proj_batch_size"]
    batches = [{k: v[i : i + S].to(args.device) for k, v in enc.items()}
               for i in range(0, enc["input_ids"].size(0), S)]

    map_rows = per_row_losses(f_seq, theta, batches)
    tol_abs = sargs["alpha_tol"] * float(map_rows.mean())
    alpha_ms, radii = measure_alpha(f_seq, theta, space, batches, pv, kdim, tol_abs)
    alphas = {"measured": alpha_ms,
              "closed_form": optimal_alpha(theta, space, kdim, n_probes=sum(n))}
    for name, a in alphas.items():
        print(f"alpha[{name:11s}] = {a:.4e}   => ||delta|| ~ {(kdim / a) ** 0.5:.4g}")
    print(f"  radii min/median/max = {min(radii):.4g} / {sorted(radii)[len(radii) // 2]:.4g} / "
          f"{max(radii):.4g}  (per part: "
          f"{[round((b['info']['kernel_dim'] / b['info']['alphas']['measured']) ** 0.5, 3) for b in blobs]})")

    methods = {}
    for name, a in alphas.items():
        methods[f"projected_{name}"] = pv / a ** 0.5
        methods[f"isotropic_{name}"] = iso / a ** 0.5
    info = dict(blobs[0]["info"])
    info.update({
        "kernel_dim": kdim, "alphas": alphas, "alpha_tol_nats": tol_abs, "radii": radii,
        "kernel_residual_max": max(b["info"]["kernel_residual_max"] for b in blobs),
        "norms": {m: float(d.norm(dim=1).mean()) for m, d in methods.items()},
        "merged_from": [str(p) for p in args.parts], "seeds": seeds,
        "timings": {k: sum(b["info"]["timings"].get(k, 0.0) for b in blobs)
                    for k in blobs[0]["info"]["timings"]},
    })
    out_args = dict(sargs, seed=seeds[0], n_samples=sum(n), out=args.out)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"methods": {m: d.float() for m, d in methods.items()},
                "names": blobs[0]["names"], "shapes": blobs[0]["shapes"],
                "info": info, "args": out_args}, out)
    out.with_suffix(".json").write_text(json.dumps(
        {"info": {k: v for k, v in info.items() if k != "batch_ranks"}, "args": out_args}, indent=2))
    print(f"saved {len(methods)} methods x {sum(n)} samples to {out}")


if __name__ == "__main__":
    main()
