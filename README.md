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

# 2. Draw from q(theta) = N(theta_map, alpha^-1 U_L U_L^T), once per loss mode.
#    --split MUST match stage 1: we project against theta_map's own training data.
#    Saves projected and isotropic samples at a *measured* alpha and at Lemma
#    3.4's alpha*, plus diagonal / full linearised Laplace (exact path only).
uv run python scripts/sample_posterior.py --loss-mode sequence
uv run python scripts/sample_posterior.py --loss-mode token --n-samples 8

# 3. Every method on train (Lemma 4.3), held-out test and OOD text: BMA NLL,
#    accuracy, ECE, entropy, mutual information, OOD AUROC.
uv run python scripts/evaluate_posterior.py

# 4. Baselines side by side, with timings.
uv run python scripts/compare_methods.py --adapter checkpoints/smollm2_lora

# 5. Six-panel diagnostic figure (correlation heatmaps, spectra, Lemma 4.3 curve).
uv run python scripts/visualize_posterior.py --adapter checkpoints/smollm2_lora --loss-mode sequence
```

On DTU HPC: `sh scripts/hpc/submit_pipeline.sh` submits stages 1-3 as four
dependent LSF jobs (the two loss modes sample in parallel). `SKIP_FINETUNE=1`
reuses an existing `checkpoints/smollm2_lora`.

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
| diagonal Laplace | 0.08 s | 4.1e-02 | mean field; not in the kernel |
| linearised Laplace (full cov) | 0.08 s | 3.9e-02 | exact via SVD, no KFAC needed |
| **projected, exact SVD** | **0.09 s** | **4.3e-08** | machine precision |
| projected, alternating | 113 s | 4.7e-05 | matrix-free, scales to any `R` |

Both projected routes produce the same `||delta||` (26.13 vs 26.13), which is an
independent check on the iterative implementation. But the exact route is
**~1000x faster here**, so:

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

Lemma 3.4's proof (Appendix A.3) maximises the approximate log evidence

```
log q(D | alpha) ~ -alpha ||theta_map||^2 / 2 + (rank / 2) log(alpha) + C     (Eq. 40)
```

with `rank = P - Tr(I - P(GGN)) = rank(J^L)`. Setting the derivative to zero
(Eq. 41) gives

```
alpha* = rank(J^L) / ||theta_map||^2
```

**The paper prints the reciprocal** (Eq. 9 and Eq. 42:
`alpha* = ||theta_map||^2 / rank`), which does not solve its own Eq. 41. The
corrected form is MacKay's evidence update `alpha = gamma / ||w||^2`, with
`gamma` the number of well-determined parameters. An earlier version of this
repository used the printed form; `tests/test_projection.py` now checks that
`optimal_alpha` maximises Eq. 40. (The variance corollary of Lemma 4.3 has a
similar slip: `Var <= O(alpha^2)` should be `O(alpha^-2)`.)

Since the covariance is `alpha^-1 U U^T` with `U U^T` a projector of rank
`P - rank`, the typical sample size is

```
E||delta||^2 = alpha^-1 * tr(U U^T) = alpha^-1 * (P - rank)
             = ||theta_map||^2 * (P - rank) / rank
```

Two things follow:

1. `||delta||` **falls monotonically with the number of rows**, and vanishes as
   `rank -> P`.
2. In the regime `rank << P` -- the regime every LoRA setup is in --
   `||delta|| ~ ||theta_map|| * sqrt(P / rank)`: a sample is `sqrt(P / rank)`
   times larger than `theta_map` itself.

Measured on the smoke test (`P = 2048`, exact `rank(J^L) = 16`,
`||theta_map||^2 = 5.36`): `alpha* = 2.99`, giving `||delta|| = 26`. That is
still far outside the linear regime, where Lemma 4.3's `O(||delta||^2)`
guarantee is worthless: projected and unprojected samples then degrade the loss
about equally. (`scripts/smoke_test.py` prints that row beside a second one
computed from the *estimated* kernel dimension, which gives `alpha* = 1.56` and
`||delta|| = 36` from the same run. That gap is not rounding -- see below.)

That prediction is not just algebra: `figures/compare_methods.json`, recorded by
`scripts/compare_methods.py` -- which takes the rank from the exact SVD
(`n_rows: 16`) rather than from probes -- has `alpha: 2.986` and a *measured*
sample norm of `26.13`, from both the exact and the iterative route.

**"Optimal" does not mean "safe".** `alpha*` maximises the evidence of the
*linearised* model, in which a step along the kernel leaves every training
prediction unchanged however long it is. The evidence therefore cannot see
where the linear regime ends, and nothing in it penalises a large `||delta||`.

### What actually sets the scale

LoRA's `||theta_map||^2` is almost entirely `lora_A` -- `lora_B` is initialised
to zero and stays near it -- and Kaiming init makes `||lora_A||^2` grow in
proportion to `P`, so `c = ||theta_map||^2 / P` is a constant of the init. In
the `rank << P` regime the identity becomes

```
||delta|| ~ P * sqrt(c / rank)
```

| LoRA `r` | `P` | `||theta_map||^2` | per-param `c` | `alpha*` at `rank=16` | `||delta||` at `rank=16` |
|---:|---:|---:|---:|---:|---:|
| 2 | 1,024 | 2.81 | 0.0027 | 5.69 | 13.3 |
| 4 | 2,048 | 5.36 | 0.0026 | 2.99 | 26.1 |
| 8 | 4,096 | 10.90 | 0.0027 | 1.47 | 52.7 |
| 16 | 8,192 | 21.49 | 0.0026 | 0.74 | 104.8 |
| 32 | 16,384 | 42.61 | 0.0026 | 0.38 | 208.8 |

(Smoke-test config: 2 layers, `q_proj`/`v_proj`, at initialisation; the norms
are measured, the last two columns follow from them.) `||delta||` grows
**linearly in `P`** at fixed rank, so a smaller adapter helps proportionally --
and more rows help as `1/sqrt(rank)`.

### The estimator degenerates before the formula does

`alpha*` is proportional to `P - kernel_dim`, and `kernel_dim` comes from a
Hutchinson probe whose standard deviation is `sqrt(2 * kernel_dim / k)` on `k`
probes. The numerator is therefore a small difference between two large, noisy numbers.
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
returns `alpha = 1 / ||theta_map||^2`.

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
2. **Use more projection rows: `mode="token"`.** It multiplies the rank by `T`,
   which raises `alpha*` by `T` and shrinks `||delta||` by about `sqrt(T)` --
   *and* tightens the guarantee from each sequence's mean loss to every
   token's loss. On the smoke test, token mode has rank 368 instead of 16,
   giving `alpha* = 68.7` and `||delta|| ~ 4.9` instead of 26. Statistically
   there is no trade-off; the cost is compute. The dense `J^L` grows by `T`
   and usually stops fitting, which forces the iterative projection, and that
   can stall on correlated rows (adjacent tokens are exactly that) -- see the
   convergence section.
3. **Shrink the adapter.** `||delta||` is linear in `P` at fixed rank (table
   above).
4. **Take `rank` from the exact SVD when the dense `J^L` fits.** That removes
   the cancellation above, though not the scale problem.

This is not a flaw in the method's design, beyond the misprint: it targets
`P >> N*O` with `O` large, so `rank = N*O` is substantial. A causal LM in
`mode="sequence"` collapses `O` to 1, the smallest rank -- and so the largest
`||delta||` -- available. Worth writing up.

## Open choice: what is "one datum" for a sequence model?

`make_causal_lm_loss(mode=...)` implements both:

- `"sequence"` — one row per sequence (mean NLL over its tokens). Matches Eq. 14
  literally; `J^L` is `(N, P)`. **Start here.**
- `"token"` — one row per token; `J^L` is `(N·T, P)`. Closer to the full-Jacobian
  kernel of §4 and a stronger per-token guarantee, but `T×` the cost, a dense
  `J^L` far too large to factor directly. Under Lemma 3.4 it also gives a
  *smaller* `||delta||` (alpha section above).

The paper does not address sequence models, so this trade-off is unexplored and
is a plausible contribution in itself.

## Evaluation plan

- **Lemma 4.3 / no underfitting** — worst change in any training sequence's
  mean NLL under posterior samples, against an unprojected `N(0, α^-1 I)`
  control at identical scale. `scripts/evaluate_posterior.py` reports it as
  `trainΔ`; it is the headline result.
- **Held-out predictive quality** — BMA NLL, accuracy and ECE on Alpaca
  `train[2000:2256]`, which fine-tuning never sees.
- **Lemma 3.2 / OOD variance** — mutual information in-distribution vs
  WikiText-2, and the AUROC it gives for OOD detection.
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
