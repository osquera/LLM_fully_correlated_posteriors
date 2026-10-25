"""Non-Bayesian-posterior baselines, as a samples file the evaluation scripts read
like any posterior (offsets delta from theta_map in the same LoRA ParamSpace):

  lora_ensemble  M LoRA adapters fine-tuned like theta_map with other seeds
                 (finetune_lora.py --seed s): a deep ensemble of the adapter,
                 the "if compute were free" reference. delta_m = theta_m - theta_map,
                 so theta_map + delta_m is member m exactly.
  dropconnect    MC-dropout-style: every LoRA weight dropped with probability p
                 and the rest scaled by 1/(1-p), one fixed mask per sample
                 (dropout on the weights, since the adapter was trained without
                 activation dropout).

Neither is scaled to a matched norm: each is used as it would be in practice.
The file copies the training-data args of --like (a token-mode samples file),
so evaluate_posterior.py and analyze_posterior.py accept it alongside it.

  uv run python scripts/build_baseline_samples.py \\
      --members checkpoints/smollm2_lora_ens{1..8} --like samples/posterior_token.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from llmproj.paramspace import ParamSpace


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora", help="theta_map")
    ap.add_argument("--members", nargs="*", default=[], help="ensemble adapters")
    ap.add_argument("--like", default="samples/posterior_token.pt",
                    help="samples file whose training-data args to copy")
    ap.add_argument("--dropconnect-p", type=float, nargs="*", default=[0.1])
    ap.add_argument("--n-samples", type=int, default=8, help="DropConnect masks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="samples/baselines.pt")
    return ap.parse_args()


def lora_theta(base_model: str, adapter: str) -> tuple[ParamSpace, torch.Tensor]:
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter)
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    return space, space.flatten(space.theta(model)).detach().clone()


def main():
    args = parse_args()
    like = torch.load(args.like, weights_only=False)
    space, theta = lora_theta(args.base_model, args.adapter)
    print(f"theta_map from {args.adapter}: {space}")

    methods = {}
    if args.members:
        deltas = []
        for m in args.members:
            sp, th = lora_theta(args.base_model, m)
            if sp.names != space.names or sp.shapes != space.shapes:
                raise SystemExit(f"{m}: LoRA layout differs from {args.adapter}")
            deltas.append(th - theta)
        methods["lora_ensemble"] = torch.stack(deltas)
    gen = torch.Generator().manual_seed(args.seed)
    for p in args.dropconnect_p:
        keep = (torch.rand(args.n_samples, space.P, generator=gen) >= p).float()
        methods[f"dropconnect_p{p:g}"] = theta * (keep / (1.0 - p)) - theta

    for m, d in methods.items():
        print(f"  {m:20s} {d.shape[0]} samples, mean ||delta|| = {float(d.norm(dim=1).mean()):.4g}")
    info = {"P": space.P, "loss_mode": "baseline", "loss_on": like["info"].get("loss_on", "all"),
            "alpha_tol_nats": like["info"]["alpha_tol_nats"],
            "members": args.members, "dropconnect_p": args.dropconnect_p,
            "norms": {m: float(d.norm(dim=1).mean()) for m, d in methods.items()}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"methods": methods, "names": space.names, "shapes": space.shapes,
                "info": info, "args": {**like["args"], "baseline_args": vars(args)}}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
