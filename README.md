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
# 1. theta_map: LoRA fine-tune SmolLM2-135M. theta = adapters only (P = 460,800).
uv run python scripts/finetune_lora.py --model HuggingFaceTB/SmolLM2-135M

# 2. Draw from q(theta) = N(theta_map, alpha^-1 U_L U_L^T).
#    --split MUST match stage 1: we project against theta_map's own training data.
uv run python scripts/sample_posterior.py --n-samples 8 --n-iterations 500

# 3. Test the paper's claims against an unprojected control.
uv run python scripts/eval_underfitting.py

# 4. Baselines side by side, with timings.
uv run python scripts/compare_methods.py --adapter checkpoints/smollm2_lora

# 5. Six-panel diagnostic figure (correlation heatmaps, spectra, Lemma 4.3 curve).
uv run python scripts/visualize_posterior.py --adapter checkpoints/smollm2_lora --loss-mode sequence
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

The defaults here are SmolLM2-135M with `r=8` on `q_proj`/`v_proj`, i.e.
`P = 30 * 8 * (1152 + 768) = 460,800`. For `mode="sequence"` with `N=256`, dense
`J` is ~0.47 GB: fits. Under `mode="token"` with `T=256`, `R` becomes 65k and
dense `J` is ~121 GB: does not fit, and the alternating projection is the only
option. That is exactly the trade-off the paper's algorithm exists to solve.

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
denominator is `rank(J^L)`. Since the covariance is `alpha^-1 U U^T` with
`U U^T` a projector of rank `P - rank`, the typical sample size is

```
E||delta||^2 = alpha^-1 * tr(U U^T) = alpha^-1 * (P - rank)
             = rank * (P - rank) / ||theta_map||^2
```

Two things follow:

1. `||delta||` peaks at `rank = P/2` and vanishes as `rank -> P`.
2. **In the regime `rank << P` -- which is the regime every LoRA setup is in --
   `||delta||` GROWS with rank**, roughly as `sqrt(rank * P) / ||theta_map||`.

Point 2 is what an earlier version of this README got backwards, and it is worth
being precise about why. The error was *not* the `P` versus `P - rank`
distinction: when `rank << P` those two agree to well under a percent, and
either one gives `||delta|| ~ sqrt(rank * P) / ||theta_map||`. The error was a
sign slip on the closed form itself. The old text claimed `mode="token"`
*raises* `alpha*` by a factor of `T`; since `alpha* = ||theta_map||^2 / rank`,
multiplying the rows by `T` *divides* `alpha*` by `T`. More rows, bigger
posterior.

Measured on the smoke test (`P = 2048`, exact `rank(J^L) = 16`,
`||theta_map||^2 = 5.36`): `alpha* = 0.33`, giving `||delta|| = 78`. That is far
outside the linear regime, where Lemma 4.3's `O(||delta||^2)` guarantee is
worthless: projected and unprojected samples then degrade the loss about
equally. (`scripts/smoke_test.py` prints that row beside a second one computed
from the *estimated* kernel dimension, which gives `alpha* = 0.64` and
`||delta|| = 56` from the same run. That gap is not rounding -- see below.)

That prediction is not just algebra: `figures/compare_methods.json`, recorded by
`scripts/compare_methods.py` -- which takes the rank from the exact SVD
(`n_rows: 16`) rather than from probes -- has `alpha: 0.3349` and a *measured*
sample norm of `78.02`, from both the exact and the iterative route.

### What actually sets the scale

In the `rank << P` regime the identity rearranges to

```
||delta|| ~ sqrt( rank / (||theta_map||^2 / P) )
```

so only two quantities matter: the number of loss rows, and the *per-parameter*
mean square of `theta_map`. `P` itself cancels. That is not an asymptotic
argument, it is measurable. LoRA's `||theta_map||^2` is almost entirely
`lora_A` -- `lora_B` is initialised to zero and stays near it -- and Kaiming
init makes `||lora_A||^2` grow in proportion to `P`:

| LoRA `r` | `P` | `||theta_map||^2` | per-param | `||delta||` at `rank=16` |
|---:|---:|---:|---:|---:|
| 2 | 1,024 | 2.81 | 0.0027 | 75.8 |
| 4 | 2,048 | 5.36 | 0.0026 | 77.9 |
| 8 | 4,096 | 10.90 | 0.0027 | 77.4 |
| 16 | 8,192 | 21.49 | 0.0026 | 78.0 |
| 32 | 16,384 | 42.61 | 0.0026 | 78.4 |

(Smoke-test config: 2 layers, `q_proj`/`v_proj`, at initialisation.) A sixteen-
fold change in `P` moves `||delta||` by three percent. **Shrinking the adapter
does not help** -- the per-parameter scale is set by the LoRA init, not by the
parameter count. The driver is not `P >> N`, and not `P` at all: it is that
`theta_map` is small *per parameter*, which for LoRA is structural.

### The estimator degenerates before the formula does

`alpha*` divides by `P - kernel_dim`, and `kernel_dim` comes from a Hutchinson
probe whose standard deviation is `sqrt(2 * kernel_dim / k)` on `k` probes. The
denominator is therefore a small difference between two large, noisy numbers.
A single smoke-test run yields three estimates of the same rank:

| source | kernel dim | implied rank |
|---|---:|---:|
| exact, `sum(a.rank for a in fac)` | 2032 | **16** |
| `estimate_kernel_dim`, 4 probes | 2040 | 8 |
| sampler's `kernel_dim_est` | 2005.7 | 42 |

The predicted probe std is `sqrt(2 * 2032 / 4) = 32`, so `rank = 16 +/- 32`:
all three readings are statistically consistent, and all three are useless. At
the target scale it is worse, not better. With `P = 460,800`, `rank = N = 256`
and 8 probes the std is `sqrt(2 * 460,544 / 8) = 339`, so the estimate can come
out negative -- at which point `optimal_alpha` clamps with `max(rank, 1.0)` and
returns `alpha = ||theta_map||^2`.

Relative accuracy on `kernel_dim` is excellent and entirely beside the point.
What `alpha*` needs is relative accuracy on `P - kernel_dim`, and that is a
catastrophic cancellation. `optimal_alpha` now warns when the estimated rank
falls within `3 sigma` of zero; pass `n_probes` to enable the check.

### Ways out

The projection itself is sound; the *scale* is what breaks. Verified at matched
`||delta||` on the tiny Llama, projected vs isotropic direction:

| `||delta||` | projected | unprojected | ratio |
|---:|---:|---:|---:|
| 1e-1 | 5.5e-05 | 2.2e-03 | **39x** |
| 1e-2 | 9.5e-07 | 2.2e-04 | **228x** |
| 1e-3 | 9.5e-07 | 2.2e-05 | 23x |

The projected column floors at ~1e-6, i.e. fp32 resolution: the loss is
preserved to machine precision. The ratio shrinks at 1e-3 only because of that
floor, not because the projection got worse. So, in order of preference:

1. **Set `--alpha` from a measurement, not from Lemma 3.4.** Sweep `||delta||`
   on the real model, find where the worst per-datum loss change exceeds your
   tolerance, and pick `alpha` to land inside that. This is the only method here
   that reliably works, and it is cheap: one kernel direction and a line search.
2. **Use fewer projection rows, not more.** Counter-intuitive, but it follows
   from the identity above: in the `rank << P` regime, adding rows inflates the
   posterior. It is not free -- fewer rows means a larger kernel, so fewer of
   the sampled directions are genuinely loss-neutral and the Lemma 4.3
   guarantee covers less. This buys scale at the cost of the guarantee rather
   than improving both.
3. **Take `rank` from the exact SVD when the dense `J^L` fits.** That removes
   the cancellation above, though not the scale problem.

**Do NOT reach for `mode="token"` to fix this.** It multiplies rank by `T`,
which makes `||delta||` larger, and it forces you onto the iterative projection
(dense `J^L` no longer fits) which can stall -- see the convergence section.
An earlier version of this README recommended exactly that, wrongly.

This is not a flaw in the paper: it targets `P >> N*O` with `O` large, so
`rank = N*O` is substantial and `||theta_map||` is a full network's worth of
trained weights, large per parameter. A causal LM with LoRA collapses `O` to 1
*and* pins the per-parameter scale at its initialisation value, which is where
the closed form degenerates. Worth writing up.

## Open choice: what is "one datum" for a sequence model?

`make_causal_lm_loss(mode=...)` implements both:

- `"sequence"` — one row per sequence (mean NLL over its tokens). Matches Eq. 14
  literally; `J^L` is `(N, P)`. **Start here.**
- `"token"` — one row per token; `J^L` is `(N·T, P)`. Closer to the full-Jacobian
  kernel of §4 and a stronger per-token guarantee, but `T×` the cost, a dense
  `J^L` far too large to factor directly, and — per the alpha section above — a
  *larger* `||delta||` under Lemma 3.4, not a smaller one.

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
