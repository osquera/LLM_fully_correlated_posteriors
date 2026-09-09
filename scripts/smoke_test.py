"""End-to-end smoke test on a tiny randomly-initialised Llama (SmolLM2's family).

No download, no training: verifies that functional_call + torch.func JVP/VJP
compose with a HuggingFace Llama forward pass, that LoRA parameters can be
isolated as theta, and that Lemma 4.3 holds on real transformer weights.
"""
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from llmproj.alpha import estimate_kernel_dim, optimal_alpha
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace
from llmproj.projection import (
    kernel_residual,
    precompute_batch_pinv,
    project_vector,
    sample_projected_posterior,
)

torch.manual_seed(0)
V, T, S, N_SEQ = 256, 24, 4, 16

cfg = LlamaConfig(vocab_size=V, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=4, max_position_embeddings=T,
                  attn_implementation="eager")
model = LlamaForCausalLM(cfg).to(torch.float32)
model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8,
                                         target_modules=["q_proj", "v_proj"],
                                         lora_dropout=0.0, bias="none",
                                         task_type="CAUSAL_LM"))
model.eval()

space = ParamSpace.from_module(model, include=["lora_A", "lora_B"])
theta, frozen = space.theta(model), space.frozen(model)
print(space)

ids = torch.randint(0, V, (N_SEQ, T))
batches = [{"input_ids": ids[i:i+S], "attention_mask": torch.ones(S, T, dtype=torch.long)}
           for i in range(0, N_SEQ, S)]

for mode in ("sequence", "token"):
    f = make_causal_lm_loss(model, space, frozen, mode=mode)
    rows = f(theta, batches[0]).numel()
    print(f"\n--- mode={mode}: {rows} J^L rows per batch of {S} ---")
    fac = precompute_batch_pinv(f, theta, space, batches, progress=False)
    print(f"  per-batch ranks: {[a.rank for a in fac]}")

    eps = space.randn()
    for n in (1, 50, 300):
        v = project_vector(eps.clone(), f, theta, space, batches, fac, n_iterations=n)
        print(f"  n_iter={n:4d}  kernel residual = "
              f"{kernel_residual(v, f, theta, space, batches):.4e}")

    if mode == "sequence":
        kdim = estimate_kernel_dim(f, theta, space, batches, fac, n_probes=4, n_iterations=300)
        alpha = optimal_alpha(theta, space, kdim)
        print(f"  Lemma 3.4: kernel_dim ~ {kdim:.0f}/{space.P}  alpha* = {alpha:.4e}")

        deltas, info = sample_projected_posterior(
            f, theta, space, batches, n_samples=2, alpha=alpha,
            n_iterations=300, tol=None, progress=False)
        print("  sampling info:", json.dumps(
            {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in info.items() if k != "batch_ranks"}))

        # Lemma 4.3 must be checked at MATCHED perturbation norm. Comparing a
        # projected sample against an unprojected one drawn at the same alpha is
        # uninformative when alpha is small: both leave the linear regime, and
        # the O(||delta||^2) bound then says nothing. So fix ||delta|| and vary it.
        def dloss(d):
            pert = {k: theta[k] + dd for k, dd in space.unflatten(d).items()}
            return max(float((f(pert, bb) - f(theta, bb)).abs().max()) for bb in batches)

        vdir = project_vector(space.randn(), f, theta, space, batches, fac, n_iterations=300)
        vdir = vdir / vdir.norm()
        udir = torch.randn(space.P)
        udir = udir / udir.norm()
        res = kernel_residual(vdir, f, theta, space, batches)
        print()
        print("  Lemma 4.3 at matched norm (projected dir residual %.1e):" % res)
        print("  %10s %13s %13s %8s" % ("||delta||", "projected", "unprojected", "ratio"))
        for nrm in (1e-1, 1e-2, 1e-3):
            a, c = dloss(vdir * nrm), dloss(udir * nrm)
            print("  %10.0e %13.4e %13.4e %7.1fx" % (nrm, a, c, c / a))

        total_rank = sum(a.rank for a in fac)
        print()
        print("  NOTE: rank(J^L) = %d but P = %d, so the projection removes only"
              % (total_rank, space.P))
        print("  %d of %d directions, and alpha* = ||theta||^2 / rank is correspondingly"
              % (total_rank, space.P))
        print("  small (%.3g), giving ||delta|| ~ %.0f. See README 'The alpha scaling trap'."
              % (alpha, (space.P / alpha) ** 0.5))
