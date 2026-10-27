"""Posterior samples that follow the curved set of equally good weights
(predictor-corrector continuation) instead of its tangent at theta_map.

The projected posterior moves along ker J^L(theta_map), which keeps every
token's loss only to first order: far out (‖δ‖ >~ 25 on the biographies, E21)
the samples break the fit. Here each sample walks out along its own
projected direction u (a token-mode samples file) in steps of at most
--max-step, and after every step pulls back with a full-batch L-BFGS
correction of

    R(theta) = mean_t (l_t(theta) - l_t(theta_map))^2

over the constrained tokens. grad R = 2 mean_t r_t grad l_t lies in the row
space of J^L at the current point, so the correction moves back toward the
set {l_t = l_t(theta_map) for all t} and leaves the kernel directions,
where the samples' spread lives, alone (to first order).

At each arc length in --norms the state is saved as method
``retracted<norm>`` (‖δ‖ is reported; it can differ from the arc length),
with its fit: worst and mean |per-sequence dNLL| and the share over the
sampler's tolerance. The output is a token-mode samples file that
evaluate_bio.py / analyze_posterior.py read (pass --methods token/retracted<n>).

  uv run python scripts/retract_posterior.py --samples samples/posterior_token_bio.pt \\
      --adapter checkpoints/bio_lora --norms 6 12 25 50 81 --out samples/retracted_bio.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.data import encode_texts, load_texts
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="token-mode samples file (directions + data)")
    ap.add_argument("--method", default="projected_measured")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--norms", type=float, nargs="+", default=[6, 12, 25, 50, 81])
    ap.add_argument("--max-step", type=float, default=4.0, help="largest predictor step (‖.‖)")
    ap.add_argument("--corr-iter", type=int, default=10, help="L-BFGS iterations per correction")
    ap.add_argument("--n-samples", type=int, default=None, help="default: all in the file")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()
    blob = torch.load(args.samples, weights_only=False)
    if blob["info"]["loss_mode"] != "token":
        raise SystemExit("needs a token-mode samples file")
    sargs, tol = blob["args"], blob["info"]["alpha_tol_nats"]
    U = blob["methods"][args.method].float()[: args.n_samples]
    U = U / U.norm(dim=1, keepdim=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, attn_implementation="eager",
                                                dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta, frozen = space.theta(model), space.frozen(model)
    theta_flat = space.flatten(theta).detach()
    if U.shape[1] != space.P:
        raise SystemExit(f"samples have P = {U.shape[1]}, the adapter {space.P}")
    f = make_causal_lm_loss(model, space, frozen, mode="token")

    texts = load_texts(sargs["dataset"], sargs["split"], sargs["text_field"])[: sargs["n_train"]]
    enc = encode_texts(tokenizer, texts, sargs["max_length"], loss_on=sargs["loss_on"])
    batches = []
    for i in range(0, enc["input_ids"].size(0), args.batch_size):
        L = int(enc["attention_mask"][i : i + args.batch_size].sum(1).max())
        batches.append({k: v[i : i + args.batch_size, :L].to(args.device) for k, v in enc.items()})
    n_tok = int(enc["loss_mask"][:, 1:].sum())

    def tok_losses(flat):
        th = space.unflatten(theta_flat + flat)
        return [f(th, b) for b in batches]

    with torch.no_grad():
        ref = [x.detach() for x in tok_losses(torch.zeros(space.P, device=args.device))]
    masks = [b["loss_mask"][:, 1:].reshape(-1).float() for b in batches]

    @torch.no_grad()
    def fit(flat):
        """Per-sequence mean NLL change vs theta_map."""
        d = []
        th = space.unflatten(theta_flat + flat)
        for b, r, m in zip(batches, ref, masks):
            S = b["input_ids"].shape[0]
            diff = ((f(th, b) - r) * m).view(S, -1).sum(1) / m.view(S, -1).sum(1).clamp(min=1)
            d.append(diff.cpu())
        d = torch.cat(d)
        return {"worst": float(d.abs().max()), "mean_abs": float(d.abs().mean()),
                "mean": float(d.mean()), "frac_over_tol": float((d.abs() > tol).float().mean())}

    def correct(delta):
        delta = delta.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([delta], max_iter=args.corr_iter, history_size=10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total = 0.0
            th = space.unflatten(theta_flat + delta)
            for b, r, m in zip(batches, ref, masks):
                loss = (((f(th, b) - r) * m) ** 2).sum() / n_tok
                loss.backward()
                total += float(loss.detach())
            return torch.tensor(total)

        opt.step(closure)
        return delta.detach()

    norms = sorted(args.norms)
    out_d = {n: [] for n in norms}
    log = {n: [] for n in norms}
    for j, u in enumerate(U):
        u = u.to(args.device)
        delta, arc = torch.zeros(space.P, device=args.device), 0.0
        for n in norms:
            while arc < n - 1e-9:
                step = min(args.max_step, n - arc)
                delta = correct(delta + step * u)
                arc += step
            fr = fit(delta)
            fr["norm"] = float(delta.norm())
            fr["cos_u"] = float(delta @ u / delta.norm().clamp_min(1e-12))
            out_d[n].append(delta.cpu())
            log[n].append(fr)
            print(f"sample {j} arc {n:g}: ‖δ‖ {fr['norm']:.2f} cos(δ,u) {fr['cos_u']:.3f}  "
                  f"seq dNLL worst {fr['worst']:.4f} mean|.| {fr['mean_abs']:.5f} "
                  f"over tol {fr['frac_over_tol']:.3f}  ({(time.perf_counter() - t0) / 60:.0f} min)",
                  flush=True)

    methods = {f"retracted{n:g}": torch.stack(out_d[n]) for n in norms}
    summary = {f"retracted{n:g}": {k: sum(r[k] for r in log[n]) / len(log[n]) for k in log[n][0]}
               for n in norms}
    print("\nmean over samples:")
    for m, r in summary.items():
        print(f"  {m:14s} ‖δ‖ {r['norm']:6.2f}  cos {r['cos_u']:.3f}  worst {r['worst']:.4f}  "
              f"mean|.| {r['mean_abs']:.5f}  over tol {r['frac_over_tol']:.3f}")
    info = {**blob["info"], "projector": "retracted", "from": args.samples, "method": args.method,
            "per_sample": log, "summary": summary,
            "timings": {"total": time.perf_counter() - t0}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"methods": methods, "names": space.names, "shapes": space.shapes,
                "info": info, "args": sargs, "retract_args": vars(args)}, out)
    out.with_suffix(".json").write_text(json.dumps({"info": info, "retract_args": vars(args)},
                                                   indent=2, default=str))
    print(f"saved {out} ({(time.perf_counter() - t0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
