"""Laplace-LoRA (Yang et al. 2024) as a samples file, for E43.

A Kronecker-factored (KFAC) Laplace approximation over the LoRA weights, the
standard Bayesian-LoRA baseline: per LoRA matrix W (out x in),

    F_W ~ A (x) G,  A = mean_t a_t a_t^T (layer inputs),  G = sum_t g_t g_t^T

over every non-padding position t of the training corpus, g_t the gradient
of the loss at the layer's output with labels sampled from the model's own
predictive (the Fisher, = the GGN for softmax cross-entropy; --n-mc draws).
The posterior is N(theta_map, (F + lam I)^-1) with an isotropic prior of
precision lam, chosen by the Laplace evidence over a log grid

    log p(D | lam) = const - lam/2 ||theta_map||^2 + P/2 log lam
                     - 1/2 sum_W sum_ij log(a_i g_j + lam).

With A = U_A diag(a) U_A^T and G = U_G diag(g) U_G^T, a sample is
    delta_W = U_G (Z / sqrt(g_j a_i + lam)) U_A^T,   Z standard normal.

Methods written (offsets from theta_map, like build_baseline_samples.py):
  laplace_kfac        lam by the evidence: the method as used in practice
  laplace_kfac_norm   the same draws rescaled to the mean ||delta|| of
                      --match-method in --like (the projected posterior), a
                      matched-norm comparison; evaluate_bio.py reports both
                      methods' fit cost.

  uv run python scripts/build_laplace_samples.py --adapter checkpoints/bio_err1_lora \\
      --dataset data/bio_err1/corpus.json --like samples/posterior_bio_sketch1.pt \\
      --out samples/laplace_bio1.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llmproj.data import encode_texts, load_texts
from llmproj.models import load_finetuned
from llmproj.paramspace import ParamSpace


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", required=True, help="theta_map (a LoRA adapter)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-mc", type=int, default=1, help="sampled labels per position")
    ap.add_argument("--like", required=True,
                    help="samples file (.pt, or its .json sidecar) whose args and norms to use")
    ap.add_argument("--match-method", default="projected_alternating5_measured",
                    help="method in --like whose mean ||delta|| laplace_kfac_norm takes")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def lora_modules(model, space: ParamSpace) -> dict[str, torch.nn.Module]:
    """param name in the ParamSpace -> the nn.Linear holding it."""
    mods = dict(model.named_modules())
    out = {}
    for n in space.names:
        if not n.endswith(".weight") or n[: -len(".weight")] not in mods:
            raise SystemExit(f"{n}: not the weight of a module")
        out[n] = mods[n[: -len(".weight")]]
    return out


def kfac_factors(model, mods, enc, batch_size, n_mc, device, gen):
    """{param name: (A, G, n_positions)} accumulated over the corpus."""
    A = {n: 0.0 for n in mods}
    G = {n: 0.0 for n in mods}
    store, n_pos = {}, 0

    def hook(name):
        def fwd(mod, inp, out):
            store[name] = [inp[0].detach(), None]
            out.register_hook(lambda g: store[name].__setitem__(1, g.detach()))
        return fwd

    handles = [m.register_forward_hook(hook(n)) for n, m in mods.items()]
    try:
        for i in range(0, enc["input_ids"].shape[0], batch_size):
            ids = enc["input_ids"][i: i + batch_size].to(device)
            att = enc["attention_mask"][i: i + batch_size].to(device)
            valid = enc["loss_mask"][i: i + batch_size, 1:].to(device).bool()
            pos = att.bool()
            n_pos += int(pos.sum())
            for _ in range(n_mc):
                store.clear()
                logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
                with torch.no_grad():
                    y = torch.multinomial(torch.softmax(logits[valid], -1), 1,
                                          generator=gen).squeeze(1)
                loss = torch.nn.functional.cross_entropy(logits[valid], y, reduction="sum")
                model.zero_grad(set_to_none=True)
                loss.backward()
                for n in mods:
                    a, g = store[n]
                    a, g = a[pos].double(), g[pos].double()
                    A[n] = A[n] + a.T @ a / n_mc
                    G[n] = G[n] + g.T @ g / n_mc
    finally:
        for h in handles:
            h.remove()
    return {n: (A[n] / n_pos, G[n]) for n in mods}, n_pos


def main():
    args = parse_args()
    like = (json.loads(Path(args.like).read_text()) if args.like.endswith(".json")
            else torch.load(args.like, weights_only=False, mmap=True))
    tok = AutoTokenizer.from_pretrained(args.adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, include = load_finetuned(args.base_model, args.adapter, args.device)
    if include is None:
        raise SystemExit("Laplace-LoRA needs a LoRA adapter")
    space = ParamSpace.from_module(model, include=include, trainable_only=False)
    for n, p in model.named_parameters():
        p.requires_grad_(any(k in n for k in include))
    theta = space.flatten(space.theta(model)).detach().double()
    mods = lora_modules(model, space)

    texts = load_texts(args.dataset, args.split)
    enc = encode_texts(tok, texts, args.max_length, loss_on=like["info"].get("loss_on", "all"))
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    factors, n_pos = kfac_factors(model, mods, enc, args.batch_size, args.n_mc, args.device, gen)
    eig = {}
    for n, (A, G) in factors.items():
        a, UA = torch.linalg.eigh(A)
        g, UG = torch.linalg.eigh(G)
        eig[n] = (a.clamp_min(0), UA, g.clamp_min(0), UG)
    print(f"KFAC over {len(mods)} LoRA matrices, {n_pos} positions, P = {space.P}")

    # prior precision by the Laplace evidence (the data-fit term is constant in lam)
    tn2, P = float(theta.norm() ** 2), space.P
    grid = torch.logspace(-4, 8, 121, dtype=torch.float64)
    def evidence(lam):
        logdet = sum(float(torch.log(torch.outer(g, a) + lam).sum()) for a, _, g, _ in eig.values())
        return -0.5 * lam * tn2 + 0.5 * P * math.log(lam) - 0.5 * logdet
    ev = [evidence(float(l)) for l in grid]
    lam = float(grid[int(torch.tensor(ev).argmax())])
    print(f"prior precision by the evidence: lam = {lam:.4g} (grid 1e-4 .. 1e8)")

    cpu = torch.Generator().manual_seed(args.seed + 1)
    draws = []
    for _ in range(args.n_samples):
        parts = {}
        for n, (a, UA, g, UG) in eig.items():
            Z = torch.randn(len(g), len(a), generator=cpu, dtype=torch.float64).to(UA.device)
            parts[n] = (UG @ (Z / torch.sqrt(torch.outer(g, a) + lam)) @ UA.T).float()
        draws.append(space.flatten(parts).float().cpu())
    lap = torch.stack(draws)
    methods = {"laplace_kfac": lap}
    target = like["info"].get("norms", {}).get(args.match_method)
    if target is None:
        raise SystemExit(f"{args.match_method} has no norm in {args.like}'s info")
    methods["laplace_kfac_norm"] = lap * (target / float(lap.norm(dim=1).mean()))

    for m, d in methods.items():
        print(f"  {m:20s} {d.shape[0]} samples, mean ||delta|| = {float(d.norm(dim=1).mean()):.4g}")
    info = {"P": space.P, "loss_mode": "baseline", "loss_on": like["info"].get("loss_on", "all"),
            "alpha_tol_nats": like["info"]["alpha_tol_nats"], "prior_precision": lam,
            "n_positions": n_pos, "matched_to": {args.match_method: target},
            "norms": {m: float(d.norm(dim=1).mean()) for m, d in methods.items()}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"methods": methods, "names": space.names, "shapes": space.shapes, "info": info,
                "args": {**like["args"], "laplace_args": vars(args)}}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
