# Experiment log

One entry per experiment: the question, how it was run, the numbers it
produced and what they showed. Newest entries at the bottom; the
[current picture](#current-picture) is kept up to date at the top. Raw outputs
live in `figures/` on the HPC (copies in the untracked `hpc_results/`).

## Shared setup and terms

- **Model.** SmolLM2-135M, frozen; LoRA r = 8, alpha = 16 on `q_proj` and
  `v_proj` of every layer (P = 460,800 parameters). theta_map = the adapter
  fine-tuned for 3 epochs on Alpaca `train[:2000]` (lr 1e-4, batch 16), loss
  on all tokens unless stated.
- **Posterior.** delta ~ N(0, alpha^-1 U U^T) on the LoRA weights, U spanning
  the kernel of the loss Jacobian J^L of the first 256 training sequences.
  *Sequence mode*: one row per sequence (mean NLL), rank 256. *Token mode*:
  one row per token (~25.7k non-zero rows), alternating projections,
  8 samples x 500 sweeps.
- **Measured alpha.** Largest prior scale at which no constrained sequence's
  mean NLL moves more than the tolerance (default 1% of the MAP's mean NLL,
  0.0114 nats): alpha = kernel_dim / r*^2, r* the smallest radius over the
  sample directions. Baselines (isotropic, diagonal Laplace, LLA) use the same
  alpha, so all methods have the same sample norm ‖δ‖.
- **Metrics.** *MI*: H(mean of sample predictions) - mean of their entropies,
  the samples' disagreement. *OOD AUROC*: per-sequence mean MI (MAP: entropy),
  test vs OOD set; 0.5 = chance, < 0.5 = inverted. *Error AUROC*: does the
  score rank tokens the MAP gets wrong above right ones. *Within entropy
  deciles*: the same AUROC computed inside bins of the MAP's entropy (does the
  score add anything beyond the softmax). *Held-out over tol*: share of
  training sequences never used in the projection whose mean NLL moves more
  than the tolerance (Lemma 4.3 off the projection set).
- **Test / OOD sets.** Test: Alpaca `train[2000:2256]`. OOD: WikiText-2
  (all tokens). Near-OOD in the Alpaca template: Dolly, CodeAlpaca, GSM8K,
  Spanish Alpaca. *All tokens* vs *answer tokens* scoring: see E11.

## Current picture

*(as of 2026-10-25; update when an entry changes it)*

1. **Keeps the fit.** At matched sample norm the token-projected posterior
   leaves training and test NLL at the MAP's and breaks the tolerance on 1.8%
   of never-projected training sequences; isotropic, diagonal Laplace and LLA
   ~90-100%, a LoRA ensemble 64% (E01, E03, E14).
2. **Detects unfamiliar input, where the MAP's entropy fails.** Answer tokens:
   GSM8K 0.82 vs entropy 0.30, Spanish 0.82 vs 0.70 (E11). It beats every
   other sampling method, including an 8-member LoRA ensemble (E14).
3. **But a single-pass hidden-state k-NN beats it at that** (WikiText 0.94,
   CodeAlpaca 0.89, GSM8K 0.90, Spanish 0.96; E14). As a pure input-novelty
   detector the posterior does not pay for its 8 forward passes.
4. **No help for errors or ambiguity.** MI adds nothing beyond the MAP's
   confidence for wrong tokens (E03, E11), and semantic MI only mirrors the
   answer spread (E09). Ambiguity was undetectable at 135M and 360M (E07, E10).
5. **Constraint density decides everything.** Sequence mode (256 constraints)
   and QA (2,320) fail; token mode (~14.7k) works; held-out violations fall
   ~3x per doubling of constrained data while new directions per token fall
   (E06, E10, E15).
6. **Open:** whether a posterior over the knowledge-bearing base weights
   catches what k-NN cannot (unknown facts on normal-looking questions; E17),
   and whether the top-k directions suffice (E16).

---

## E01 First full run: sequence vs token mode, matched-norm baselines

- **Date / job.** 2026-10-09; `submit_pipeline.sh` (jobs 01-04) on A100.
- **Question.** Does the projected posterior preserve the fit, and is its
  uncertainty better than unprojected noise of the same size?
- **Results** (all tokens; sequence seed 0, token seed 0):

  | method | ‖δ‖ | worst train dNLL | test NLL | ECE | WikiText AUROC | error AUROC (MI) |
  |---|---|---|---|---|---|---|
  | MAP | 0 | 0 | 1.3248 | 0.0111 | 0.995 (entropy) | 1-conf 0.874 |
  | seq / projected | 4.74 | 0.0114 | 1.3254 | 0.0119 | 0.35 | 0.699 |
  | seq / isotropic | 4.74 | 0.081 | 1.3266 | 0.0120 | 0.000 | 0.649 |
  | seq / diagonal Laplace | 4.74 | 0.049 | 1.3259 | 0.0119 | 0.002 | 0.653 |
  | seq / LLA | 4.74 | 0.059 | 1.3261 | 0.0125 | 0.001 | 0.655 |
  | token / projected | 8.25 | 0.0114 | 1.3249 | 0.0118 | 0.990 | 0.767 |
  | token / isotropic | 8.38 | 0.107 | 1.3336 | 0.0130 | 0.000 | 0.646 |

  Closed-form alpha (Lemma 3.4, corrected): alpha = 1.06 (sequence), samples
  at ‖δ‖ ~ 659 (40x ‖θ_map‖), test NLL ~12: unusable.
  Token mode: kernel residual 0.02 (target 1e-3), ~22 h projection on A100.
- **Showed.** Projection matters a lot at matched norm (5-10x less training
  damage, better test NLL); token mode >> sequence mode; nothing beats the MAP
  on NLL, ECE or error prediction. The closed-form alpha is unusable because
  the kernel is only flat to first order.

## E02 Tolerance sweep (1% -> 0.5 ... 20%)

- **Date / job.** 2026-10-12; `05_sweep_tolerance.lsf` (rescales saved samples).
- **Results** (all tokens):

  | tol | seq proj ‖δ‖ / WikiText AUROC | token proj ‖δ‖ / WikiText AUROC | token proj test dNLL | token iso test dNLL |
  |---|---|---|---|---|
  | 0.5% | 3.4 / 0.54 | 5.8 / 0.99 | +0.0000 | +0.0027 |
  | 1% | 4.7 / 0.35 | 8.2 / 0.99 | +0.0001 | +0.0087 |
  | 2% | 6.5 / 0.13 | 11.7 / 0.97 | +0.0003 | +0.028 |
  | 5% | 9.4 / 0.02 | 19.1 / 0.70 | +0.0023 | +0.116 |
  | 10% | 12.1 / 0.00 | 25.0 / 0.15 | +0.012 | +0.221 |
  | 20% | 15.6 / 0.00 | 30.2 / 0.08 | +0.035 | +0.321 |

  Baselines' OOD AUROC stays ~0.00 at every tolerance.
- **Showed.** Token mode's OOD advantage holds up to ~2% and vanishes beyond
  ~5% (nonlinear regime); projected stays far cheaper in NLL than isotropic at
  every size. The 1% row reproduces E01's alpha (20,572 vs 20,498: radius
  search rounding).

## E03 Follow-ups: MI beyond the softmax, never-projected data, near-OOD

- **Date / job.** 2026-10-18; `07_analyze.lsf` (all tokens).
- **Results.**
  - Error AUROC within MAP-entropy deciles: 0.42-0.50 for every method;
    logistic regression softmax -> softmax + MI: 0.903 -> 0.900. MI adds
    nothing beyond the softmax for wrong tokens.
  - Held-out over tol: seq projected 0.4% -> **36.3%** on unseen sequences;
    token projected 0.4% -> **1.8%**; baselines 88-100% on both.
  - Near-OOD (all tokens), token projected / MAP entropy: Dolly 0.71 / 0.74,
    CodeAlpaca 0.57 / 0.21, GSM8K 0.94 / 0.59, Spanish 0.97 / 0.81. All
    sequence-mode and baseline MI inverted (< 0.5).
- **Showed.** Sequence mode's 256 constraints protect only those 256
  sequences; token mode generalises. MI is a novelty signal, not an error
  signal. (Near-OOD numbers inflated by the shared template, see E11.)

## E04 Seed error bars, sequence mode (5 seeds)

- **Date / job.** 2026-10-18; `06_seeds.lsf`.
- **Results.** Projected vs baselines: error AUROC (MI) 0.699 +- 0.007 vs
  0.656-0.659 +- 0.007; WikiText AUROC 0.32 +- 0.04 vs ~0.003; test NLL
  1.3250 vs 1.326. Measured alpha varies ~10% across seeds (‖δ‖ 4.24 +-
  0.43): r* is a minimum over 16 directions.
- **Showed.** The projected-vs-baseline gaps are real, not seed noise.

## E05 Seed error bars, token mode (3 seeds)

- **Date / job.** 2026-10-20; `submit_token_seeds.sh` (jobs 08/09).
- **Results.** Projected: WikiText AUROC 0.988 +- 0.003, error AUROC (MI)
  0.763 +- 0.004, test NLL 1.325 +- 0.0001; isotropic 0.0002 / 0.641 /
  1.338. Kernel residual 0.018-0.022 in every seed (500 sweeps too few).
- **Showed.** Token-mode results are stable; the residual caveat is
  systematic, not bad luck.

## E06 Sequence mode with 1792 constrained sequences

- **Date / job.** 2026-10-20; `10_sequence_n1792.lsf`.
- **Results** (all tokens): held-out over tol 36.3% (n=256) -> 24.0%
  (n=1792, tested on train[1792:2000]) vs token 1.8%; WikiText AUROC 0.32 ->
  0.77 vs token 0.99; near-OOD 0.52 / 0.53 / 0.57 / 0.89 vs token 0.71 /
  0.57 / 0.94 / 0.97.
- **Showed.** More constraints help sequence mode on every metric, but token
  mode with 7x fewer sequences (and ~8x more constraint directions) stays
  better. Cannot separate "token-level" from "more constraints".

## E07 Semantic uncertainty, sampled answers (135M, Alpaca adapter)

- **Date / job.** 2026-10-20; `11_semantic.lsf`. 200 questions each of
  AmbigQA, TriviaQA, NQ-open; 8 samples x 10 answers, T = 0.7, 5-shot;
  NLI clustering (DeBERTa-v3-base) and MiniLM embeddings.
- **Results.** MAP accuracy 5.5% / 14% / 6%; 45-53 distinct meanings of 80.
  Wrong-answer AUROC, MAP semantic entropy vs token entropy: TriviaQA 0.79
  vs 0.71, NQ 0.87 vs 0.66. AmbigQA (ambiguous?): every score 0.46-0.56.
  Epistemic part (shuffle-corrected): mean ~0 +- 0.05, AUROC ~0.5 everywhere.
- **Showed.** Meaning space beats token space for errors (replicates semantic
  entropy). Ambiguity undetectable for a model that guesses everywhere.
  Sampling noise (~0.05 nats) swamps the posterior's disagreement (~0.003).

## E08 Response-only fine-tuning and projection

- **Date / job.** results copied 2026-10-25; `LOSS_ON=response` pipeline
  (a first attempt silently re-ran the default; fixed in cee2ab7).
- **Results.** 11,935 constrained tokens; alpha 99,218 -> ‖δ‖ 2.13
  (vs 8.25): worst sequence moves 50x the mean, short answers dominate the
  worst case. WikiText AUROC projected 0.725 vs isotropic 0.103; error AUROC
  0.718 vs 0.710. Flat across the 0.5-20% sweep (still linear regime).
- **Showed.** The per-sequence worst-case tolerance is outlier-driven for
  short responses. The template-free OOD number is far below 0.99, the first
  sign of E11.

## E09 Likelihood-based semantic MI (no sampling noise)

- **Date / job.** 2026-10-20/25; `12_semantic_likelihood.lsf` (built in a
  parallel session). Every model scores the MAP's sampled answers;
  importance weights give each sample's meaning distribution.
- **Results.** Raw semantic MI flags wrong answers (0.71-0.78) but **within
  aleatoric quintiles 0.42-0.65 (~chance)**, correlation(MI, aleatoric)
  0.4-0.7, projected = isotropic within 0.03. GSM8K vs TriviaQA: semantic MI
  AUROC 0.06 (135M QA) / 0.24 (360M QA): inverted, still after controlling
  for spread (0.04 / 0.17).
- **Showed.** At the answer level the posterior's MI is the answer spread
  times a constant: no separate epistemic channel.

## E10 Short-answer QA fine-tuning, 135M and 360M

- **Date / job.** 2026-10-25 copy; `scripts/hpc/qa/` (parallel session):
  LoRA fine-tuned on TriviaQA (response loss), token mode, 2,320 constrained
  tokens.
- **Results.** TriviaQA accuracy 14.5% (135M) / 26.5% (360M). MAP semantic
  entropy, wrong answers: TriviaQA 0.86 / 0.88, NQ 0.97 / 0.89; AmbigQA 0.50 /
  0.47. Token OOD (WikiText vs TriviaQA) projected / isotropic / MAP: 0.74 /
  0.60 / 0.86 (135M), 0.34 / 0.26 / 0.92 (360M).
- **Showed.** A larger model makes semantic entropy a strong error detector
  but still does not see ambiguity. With ~2.3k constraints on 2.4-4.3M LoRA
  parameters the posterior's OOD advantage disappears.

## E11 Answer-token-only scoring (the template confound)

- **Date / job.** 2026-10-25; `13_answer_only.lsf` (`--score-on response`).
- **Results.**

  | WikiText AUROC | all tokens | answer tokens |
  |---|---|---|
  | MAP entropy | 0.995 | 0.937 |
  | token projected | 0.990 | **0.868** |
  | token isotropic | 0.000 | 0.294 |
  | seq projected | 0.32 | 0.665 |

  Near-OOD, answer tokens, token projected / MAP entropy: Dolly 0.472 /
  0.475*, CodeAlpaca 0.497 / 0.167, GSM8K 0.820 / 0.297, Spanish 0.820 /
  0.696. Error AUROC (MI): projected 0.724, isotropic 0.708 (was 0.767 /
  0.646); MAP 1-conf 0.835.
  *Dolly biased low: 41 of 256 answers are cut off by max_length and were
  scored 0 (fixed in dbb42ac; rerun pending).
- **Showed.** The shared prompt template inflated every AUROC by 0.1-0.25
  and created the projected-vs-isotropic error-AUROC gap. What survives: a
  real OOD signal where entropy fails (GSM8K, Spanish). Report answer-token
  numbers from here on.

## E12 H100 reproducibility

- **Date / job.** 2026-10-25; `QUEUE=gpuh100` pipeline.
- **Results.** Fine-tune losses 1.9883 / 1.3466 / 1.3047 and MAP NLL 1.1388
  identical to the A100; sequence alpha 20,572 (= the current code on A100,
  E02).
- **Showed.** Same theta_map on both GPUs; all earlier samples stay valid.

## E13 Batched projection (vmap over samples and J J^T rows)

- **Date / job.** 2026-10-25; `14_benchmark_projection.lsf` (709020e).
- **Results** (H100, extrapolated to a full token run): precompute 2.36 h ->
  0.59 h (row_batch 16, 26 GB), projection 13.3 h -> 2.8 h (vec_batch 8,
  16.5 GB); real run: 2.9 h projection (A100, old code: ~22 h). Equal to the
  one-at-a-time algorithm up to round-off (tests).
- **Showed.** The old code left the GPU idle; ~4.6x from batching, ~7x vs the
  original A100 runs.

## E14 Baselines: LoRA ensemble, DropConnect, hidden-state distances

- **Date / job.** 2026-10-25; `15_baselines.lsf`; answer tokens.
- **Results.**

  | score | passes | WikiText | CodeAlpaca | GSM8K | Spanish | held-out over tol |
  |---|---|---|---|---|---|---|
  | MAP entropy | 1 | 0.937 | 0.167 | 0.297 | 0.696 | - |
  | token projected (MI) | 8 | 0.868 | 0.497 | 0.820 | 0.820 | 1.8% |
  | token isotropic (MI) | 8 | 0.294 | 0.344 | 0.347 | 0.514 | 99.9% |
  | LoRA ensemble, 8 seeds (MI) | 8 | 0.276 | 0.439 | 0.496 | 0.575 | 63.8% |
  | DropConnect p=0.1 (MI) | 8 | 0.333 | 0.309 | 0.376 | 0.510 | 18.6% |
  | k-NN, layer 15 | 1 | 0.938 | 0.888 | 0.895 | 0.957 | - |
  | Mahalanobis, layer 15 | 1 | 0.914 | 0.895 | 0.876 | 0.910 | - |
  | k-NN / Mahalanobis, layer 30 | 1 | 0.83 / 0.65 | 0.40 / 0.75 | 0.18 / 0.25 | 0.92 / 0.95 | - |

  Error AUROC (test + near-OOD): MAP 1-conf 0.865; all sampling methods raw
  0.77-0.79, within entropy deciles 0.53-0.56; hidden-state distances 0.40-0.50.
  Ensemble members are ‖δ‖ = 22.1 apart; DropConnect 5.2.
- **Showed.** The projected posterior is the best of all sampling-based
  methods, including the deep-ensemble reference (whose members disagree most
  in-distribution). But one-pass mid-layer k-NN is the better input-novelty
  detector. The posterior's case must rest on what input distance cannot see:
  unknown facts on normal-looking inputs (E17). Dolly omitted (see E11 note).

## E15 Scaling curve: constrained sequences 64 -> 1024 (partial)

- **Date / job.** 2026-10-25; `16_scaling.lsf`; same 976 held-out sequences
  (train[1024:2000]) for every n; answer tokens; at own alpha and rescaled to
  ‖δ‖ = 8.25.
- **Results** (matched norm 8.25):

  | n constrained | 64 | 128 | 256 | 512 | 1024 |
  |---|---|---|---|---|---|
  | held-out over tol | 16.4% | 5.4% | 1.8% | pending | pending |
  | held-out mean abs dNLL | 0.0038 | 0.0030 | 0.0025 | | |
  | rank (constraint directions) | ~4,600 | ~8,200 | ~14,700 | | |
  | rank per constrained token | 72% | 64% | 57% | | |
  | WikiText / GSM8K / Spanish AUROC | 0.81 / 0.74 / 0.79 | 0.84 / 0.79 / 0.81 | 0.87 / 0.82 / 0.82 | | |
  | projection time (H100) / residual | 0.8 h / 0.005 | 1.6 h / 0.010 | 2.9 h / 0.018 | | |

- **Showed so far.** Held-out violations fall ~3x per doubling (~n^-1.6);
  new directions per token fall (constraints increasingly redundant); OOD
  detection improves steadily. At fixed 500 sweeps the residual grows with n.

---

## Pending

- **E15 (rest).** n = 512 running, 1024 queued (~16 h). Prediction from the
  trend: ~0.6% and ~0.2% held-out over tol.
- **E14 rerun.** Corrected Dolly AUROCs (dbb42ac), with the batched token
  posterior.
- **E16 Top-k vs exact kernel** (`17_topk.lsf`). Remove only the top 16 ...
  2048 directions of J^L; energy share per k; held-out and OOD vs exact.
- **E17 Posterior over the base weights, PopQA.** Transformer blocks of
  SmolLM2-135M (embeddings excluded: rare-token confound), constrained on a
  FineWeb-Edu sample; does MI rise for rare entities beyond entropy and
  beyond hidden-state k-NN?
