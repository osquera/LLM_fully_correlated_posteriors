# Fully correlated projected posteriors for small LLMs

Applying **"Bayes without Underfitting: Fully Correlated Deep Learning Posteriors
via Alternating Projections"** (Miani, Roy & Hauberg, [arXiv:2410.16901](https://arxiv.org/abs/2410.16901))
to SmolLM2.

Reference implementation (JAX/Flax): https://github.com/h-roy/projected-bayes

## Why the loss-projection variant, and not the main method

The paper's headline method (§4) projects onto `ker(J_θ)` and needs a per-batch
`SO × SO` inverse, where `O` is the model's output dimension. For a causal LM,
`O = seq_len × vocab_size` - about **25M for a single 512-token SmolLM2
sequence**. That inverse is not computable.

§4.1 exists for exactly this case. It replaces the full Jacobian with the
**loss-Jacobian** (Eq. 14), one row per datum instead of `O` rows:

```
J^L_θ = [∇_θ l(f(θ,x₁),y₁); … ; ∇_θ l(f(θ,x_N),y_N)] ∈ R^{N×P}
```

Lemma 4.2 gives `ker(J_θ) ⊆ ker(J^L_θ)`, so the guarantee weakens from
*predictions preserved* to *per-datum loss preserved* (Lemma 4.3) — which is
what we actually need to claim "does not underfit".

## Install

```bash
uv sync --extra cpu     # or: uv sync --extra gpu
uv run pytest -q
```

## Pipeline

```bash
# 1. theta_map: LoRA fine-tune SmolLM2-135M. theta = adapters only (P ~ 1-5M).
uv run python scripts/finetune_lora.py --model HuggingFaceTB/SmolLM2-135M

# 2. Draw from q(theta) = N(theta_map, alpha^-1 U_L U_L^T).
#    --split MUST match stage 1: we project against theta_map's own training data.
uv run python scripts/sample_posterior.py --n-samples 8 --n-iterations 500

# 3. Test the paper's claims against an unprojected control.
uv run python scripts/eval_underfitting.py

# 4. Baselines side by side, with timings.
uv run python scripts/compare_methods.py --adapter checkpoints/smollm2_lora

# 5. Six-panel diagnostic figure (correlation heatmaps, spectra, Lemma 4.3 curve).
uv run python scripts/visualize_posterior.py --adapter checkpoints/smollm2_lora --loss-mode token
```

On DTU HPC: `bsub < scripts/submit_dtu_hpc.sh` runs all of the above.

## Library

| Module | Contents |
|---|---|
| `llmproj.paramspace` | `ParamSpace` — flat ↔ dict view over the θ subnetwork |
| `llmproj.losses` | `make_causal_lm_loss` — the rows of `J^L` for a causal LM |
| `llmproj.projection` | `precompute_batch_pinv`, `project_vector`, `sample_projected_posterior`, `kernel_residual` |
| `llmproj.alpha` | Lemma 3.4 closed-form `α*` via Hutchinson |
| `llmproj.baselines` | dense `J^L`, diagonal Laplace, full-covariance `lla`, exact projector |

`J J^T` is formed matrix-free: column `j` is `J(J^T e_j)`, one VJP then one JVP,
so a batch costs `2R` passes rather than the `P` passes needed to instantiate
`J`. The projection matrix `U_L U_L^T` is never built.

## Performance: use the exact SVD path when it fits

`J^L` has only `R` rows (`R` = training rows, *not* `N*O`), so `J^T J` is low
rank and the exact kernel projector is available from one SVD of the dense `J`.
When `R * P` fits in memory this is dramatically cheaper than iterating, and
exact. Measured on the tiny Llama (`P=2048`, `R=16`, 4 samples, 300 sweeps):

| method | time | residual | notes |
|---|---:|---:|---|
| diagonal Laplace | 0.9 s | 4.0e-02 | mean field; not in the kernel |
| linearised Laplace (full cov) | 0.9 s | 3.1e-02 | exact via SVD, no KFAC needed |
| **projected, exact SVD** | **0.95 s** | **3.7e-08** | machine precision |
| projected, alternating | 716 s | 4.1e-05 | matrix-free, scales to any `R` |

Both projected routes produce the same `||delta||` (78.02 vs 78.02), which is an
independent check on the iterative implementation. But the exact route is
**~750x faster here**, so:

- `R * P` fits in memory (`build_loss_jacobian` will tell you, and refuses past
  `--max-gb`) -> use `llmproj.baselines.exact_projection_samples`.
- It does not fit -> use `sample_projected_posterior` (alternating projections).

For `mode="sequence"` with `N=256` and a LoRA `P` of 2M, dense `J` is ~2 GB:
fits. Under `mode="token"` with `T=256`, `R` becomes 65k and dense `J` is ~500 GB:
does not fit, and the alternating projection is the only option. That is exactly
the trade-off the paper's algorithm exists to solve.

The baselines are cheap because they reuse that same dense `J`; neither needs
its own pass over the data.

## Convergence: budget for this

Alternating projections converge linearly at rate `∏ cos²(θ_b)` (Lemma 4.1),
and in practice that is **slow**. Measured on a 131-parameter toy problem
against the exactly-computed projector (`tests/test_projection.py::test_converges_to_exact_projector`):

| sweeps | kernel residual | rel. error vs exact | cos angle |
|---:|---:|---:|---:|
| 1 | 1.0e-01 | 3.4e-01 | 0.9456 |
| 20 | 4.0e-02 | 2.3e-01 | 0.9745 |
| 200 | 3.2e-03 | 4.4e-02 | 0.9990 |
| 600 | 2.7e-03 | 3.1e-03 | 0.9999951 |
| 2000 | 7.6e-08 | 7.2e-07 | 1.0000000 |

It *does* reach the exact projector — the implementation is verified — but a
handful of sweeps is nowhere near converged. The reference repo's
`in_between_uncertainty.py` runs with `n_iterations=1`. **Always check
`kernel_residual` before believing a posterior sample**: at residual `1e-1`,
Lemma 4.3 simply does not hold, and the "no underfitting" guarantee is void.

von Neumann `acceleration` (default on) only helps past a few hundred sweeps
(at 200 sweeps: 2.0e-2 → 3.2e-3); below that it is a wash.

## Deliberate differences from the reference implementation

1. **Relative eigenvalue cutoff.** The reference uses a hardcoded absolute
   `1e-3` on the eigenvalues of `J J^T`. LLM loss-gradient scales differ from a
   toy MLP by orders of magnitude, so an absolute threshold silently keeps
   numerical noise or discards real directions. We use `rtol * λ_max`.
2. **No `jax_debug_nans`.** The reference sets it at import in
   `precompute_loss_inv.py` and `alternating_loss_projections.py`. Ruinous at
   this scale.
3. **`acceleration` defaults on**, and `n_iterations` defaults to 500, not 10.
4. **Early stopping** on `kernel_residual` via `tol`.
5. **PyTorch.** The algorithm needs only JVP/VJP, which `torch.func` provides.
   Porting SmolLM2 to Flax instead would mean pinning `transformers<5` (Flax
   was removed in v5) and fighting weight conversion for the whole project.

## The alpha scaling trap

Lemma 3.4 gives `alpha* = ||theta_map||^2 / (P - Tr(I - P(GGN)))`, and that
denominator is `rank(J^L)`. Under `mode="sequence"` the rank is just `N`, the
number of training sequences, so when `P >> N` the closed form returns a **very
small alpha** -- an enormous prior variance. Measured on the smoke test
(`P = 2048`, `N = 16`): `alpha* = 0.64`, giving `||delta|| = sqrt(P/alpha) ~ 57`.
That is far outside the linear regime, where Lemma 4.3's `O(||delta||^2)`
guarantee is worthless: projected and unprojected samples then degrade the loss
about equally.

The projection itself is sound; the *scale* is what breaks. Verified at matched
`||delta||` on the tiny Llama, projected vs isotropic direction:

| `||delta||` | projected | unprojected | ratio |
|---:|---:|---:|---:|
| 1e-1 | 5.5e-05 | 2.2e-03 | **39x** |
| 1e-2 | 9.5e-07 | 2.2e-04 | **228x** |
| 1e-3 | 9.5e-07 | 2.2e-05 | 23x |

The projected column floors at ~1e-6, i.e. fp32 resolution: the loss is
preserved to machine precision. The ratio shrinks at 1e-3 only because of that
floor, not because the projection got worse.

Three ways out, in order of preference:

1. **Use `mode="token"`.** Rank becomes `N*T` instead of `N`, which tightens the
   kernel *and* raises `alpha*` by a factor of `T`. This is the main practical
   argument for the per-token variant.
2. **Increase `N`** until `rank(J^L)` is a meaningful fraction of `P`.
3. **Override `--alpha`** and report a sensitivity sweep, rather than trusting
   the closed form in a regime it was not designed for.

This is not a flaw in the paper: it targets `P >> N*O` with `O` large, so
`rank = N*O` is substantial there. A causal LM under `mode="sequence"` collapses
`O` to 1, which is exactly where the closed form degenerates. Worth writing up.

## Open choice: what is "one datum" for a sequence model?

`make_causal_lm_loss(mode=...)` implements both:

- `"sequence"` — one row per sequence (mean NLL over its tokens). Matches Eq. 14
  literally; `J^L` is `(N, P)`. **Start here.**
- `"token"` — one row per token; `J^L` is `(N·T, P)`. Closer to the full-Jacobian
  kernel of §4, `T×` the cost, and a strictly smaller kernel — so less posterior
  spread, but a stronger loss-preservation guarantee.

The paper does not address sequence models, so this trade-off is unexplored and
is a plausible contribution in itself.

## Evaluation plan

- **Lemma 4.3 / no underfitting** — train-set perplexity of posterior samples vs
  MAP, against an unprojected `N(0, α^-1 I)` control at identical scale.
  `scripts/eval_underfitting.py` does this; it is the headline result.
- **Lemma 3.2 / OOD variance** — predictive entropy and sample disagreement,
  in-distribution vs OOD text.
- **Downstream** — calibration (ECE) on multiple-choice tasks, selective
  prediction, hallucination detection.
- **Baselines** — MAP, last-layer Laplace, diagonal Laplace, MC-dropout, LoRA
  deep ensemble, and **Laplace-LoRA** (Yang et al., 2024), the closest prior work.

## Gotchas

- **`attn_implementation="eager"`.** Forward-mode AD (`torch.func.jvp`) does not
  compose with fused SDPA/flash attention kernels. Both scripts set this.
- **`lora_dropout=0.0`.** The projection assumes a deterministic `f`.
- **fp32 for the projection.** `U_L U_L^T` has 0/1 eigenvalues and is
  well-conditioned, but `(J J^T)^{-1}` is not.
- **Memory.** `n_samples × P` floats must be resident. This is why θ is the
  adapters, not all 135M weights.
- **Hutchinson variance.** `estimate_kernel_dim` has std `sqrt(2*kernel_dim/n_probes)`.
  Measured against an exact kernel dimension of 91: `n_probes=4` gave
  86.1 / 91.9 / 85.0, `n_probes=64` gave 89.2 / 89.8 / 89.2. Negligible in
  *relative* terms at LLM scale, but do not trust 4 probes on a small problem.
