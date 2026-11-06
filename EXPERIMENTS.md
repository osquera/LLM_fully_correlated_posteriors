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

*(as of 2026-11-06; update when an entry changes it)*

1. **Keeps the fit.** At matched sample norm the token-projected posterior
   leaves training and test NLL at the MAP's and breaks the tolerance on 1.8%
   of never-projected training sequences; isotropic, diagonal Laplace and LLA
   ~90-100%, a LoRA ensemble 64% (E01, E03, E14).
2. **Detects unfamiliar input, where the MAP's entropy fails.** Answer tokens:
   GSM8K 0.83 ± 0.01 vs entropy 0.30, Spanish 0.82 ± 0.01 vs 0.70 (E11,
   3 posterior seeds in E18). It beats every
   other sampling method, including an 8-member LoRA ensemble (E14).
3. **But a single-pass hidden-state k-NN beats it at that** (WikiText 0.94,
   CodeAlpaca 0.89, GSM8K 0.90, Spanish 0.96; E14), also when its bank holds
   only the posterior's 256 sequences (E20). As a pure input-novelty detector
   the posterior does not pay for its 8 forward passes. It is not redundant,
   though: MI and k-NN distance are nearly uncorrelated (rho 0.1-0.2), MI
   still separates WikiText / GSM8K / Spanish within k-NN quintiles
   (0.81 / 0.66 / 0.76), and their sum edges past k-NN on those sets (E20).
4. **No help for errors or ambiguity.** MI adds nothing beyond the MAP's
   confidence for wrong tokens (E03, E11), and semantic MI only mirrors the
   answer spread (E09). Ambiguity was undetectable at 135M and 360M (E07, E10).
5. **Constraint density decides everything.** Sequence mode (256 constraints)
   and QA (2,320) fail; token mode (~14.7k) works; held-out violations fall
   >= 3x per doubling of constrained data and are ~0 from 512 sequences on,
   while OOD detection rises to ~0.87 (E06, E10, E15).
6. **The constraints are low-rank, and the OOD signal is in the top
   directions.** 2048 of ~14.7k directions carry 96% of J^L's energy and
   recover 94-100% of the exact posterior's OOD AUROCs; held-out safety at a
   given sample size needs the tail (E16). But knowledge coverage lives in
   the tail: on the biographies top-1024 (93% of the energy) falls to the
   isotropic level (E24). Top-k scales input-novelty detection, not this.
7. **It knows which facts the training data supports, where input distance
   is blind.** Synthetic biographies (E17): projected MI separates never-seen
   people 0.861 and rarely-seen facts 0.804 within MAP-entropy quintiles,
   against MAP 1-conf 0.64 / 0.67, isotropic 0.71 / 0.62 and mid-layer k-NN
   ~0.4, and it adds to the softmax in a held-out logistic test (unseen
   0.864 -> 0.909, rare 0.900 -> 0.933). But an 8-member LoRA ensemble
   (8 full fine-tunes) does better still: 0.946 / 0.955 within quintiles,
   0.975 / 0.976 logistic. Not a size effect: scaled up to the ensemble's
   distance the linear posterior breaks the fit and loses the signal, while
   the ensemble members still fit the corpus as well as theta_map (E21).
   Pulling the samples back onto the fit set (a curved posterior) repairs
   the fit 14x and removes the collapse, but the signal still peaks at
   small ‖δ‖ (E23); and a 2-member ensemble already beats the 8-sample
   posterior (E22). The ensemble's advantage is where independent runs
   land, not distance or inference cost; and at equal training compute
   (a projection ~ 25 LoRA fine-tunes even at 25 sweeps, E25; 277 as run in
   E17) the gap is far larger. On wrong answers (E28, three
   replications, ~500 errors each) the posterior beats the softmax in
   every replication (+0.08 within quintiles) and adds to it in a held-out
   logistic test where the ensemble adds ~nothing; against the ensemble
   it is +0.024 on average (+0.09 / -0.02 / 0.00 by replication). E25b's
   42-error comparison was too small to tell. Combined with the ensemble it
   adds +0.025 on errors at equal inference cost, which 8 more members do
   not (+0.002); on coverage 8 more members match it (E28b, E28c). On real PopQA facts
   (E29) it does not: the fine-tune leaves ~80 errors, mostly first-token
   misses, and the softmax (0.94) is not improved by either the posterior
   or the ensemble.
   E30 says why: the edge is on **near-miss continuations** (a swapped
   training value, or a wrong token after a right first token: post 0.780 /
   0.792 vs ens16 0.708 / 0.701), which every member makes alike, while the
   ensemble's own edge is on first-token errors (0.842 vs post 0.767), which
   it catches as novelty. The posterior also beats the softmax most where
   the fact was fit (f = 1 0.807, f = 2 0.875) and still adds to the MAP's
   confidence and to the training fit in a held-out logistic test. It does
   **not** transfer to real facts yet: on PopQA (E29 rep 2) the posterior
   is below the MAP's own confidence on errors (0.703 vs 0.816) and well
   below the ensemble on coverage (0.638 vs 0.851), though the MAP there is
   already 92-98% accurate and there are only 82 errors.
8. **The projection was over-solved.** 10 alternating sweeps (vs 1000)
   give the same samples' signal and fit (5 nearly), since the measured
   alpha absorbs the residual; with a faster alpha search the bio
   posterior costs ~50 min (~18 fine-tunes, vs 12.6 h in E17), now
   dominated by the J J^T precompute, which is near its floor on an H100.
   Block-Jacobi CG and TF32 do not help (E25, E27). **A Gaussian sketch of
   each batch's J J^T** (its numerical rank is ~20% of its rows) cuts the
   precompute ~5x with the same samples (cosine 0.999) and AUROCs (E36):
   LoRA posterior ~19 min, ~4 fine-tunes, half the 8-member ensemble.
9. **A posterior knows only what its constraints contain.** On a
   pretrained model constrained on generic text it keeps the fit (also on
   held-out text) but sees neither fact popularity nor errors; semantic
   entropy wins there (E26).
10. **LoRA is not just a prior.** Projecting over all 135M weights (E31)
   keeps the fit and the posterior's edge over the softmax on errors
   (+0.05), and rescaling the prior per tensor changes nothing; but it
   costs ~2x the LoRA posterior and the full ensemble becomes the
   (narrowly) better error detector. At lr 1e-4 (E35, E37) the full MAP
   answers more probes than LoRA, and there the full posterior loses to
   the full ensemble on every task; its error-detection edge is a LoRA
   result. With the sketch, the LoRA posterior costs ~3.3 fine-tunes and
   is the best error detector of the posterior and 2-, 4-, 8-member
   ensembles, while 2 members already beat it on coverage (E38).
11. **E38's error-detection win is narrow.** It does not hold at
   SmolLM2-360M (ens4 / ens8 ahead by 0.05-0.07, posterior ~4.4
   fine-tunes; E39), on harder PopQA (below the MAP's confidence, E40),
   and TOFU leaves no errors at 20 epochs (E42). What holds in every bio
   setting (135M, 360M) is the held-out logistic test: the posterior
   adds ~+0.02 to the softmax on wrong answers where ensemble MI adds
   ~0. On coverage the posterior's best showings are TOFU unseen
   authors (level with 8 members, ahead of 2-4) and PopQA unseen facts
   (ahead of 2 members, level with 4-8). The per-batch rank is 15-33% of
   the rows on every corpus (highest on the underfit PopQA model), and
   the sketch needs k above it (E41). Cost in fine-tunes depends on the
   fine-tune: 3.3 (bio 135M) to ~18 (6-epoch PopQA).
12. **Laplace-LoRA is not a competitor here.** KFAC Laplace over the LoRA
   weights with an evidence-tuned prior (E43) is dominated by the prior
   (||delta|| = sqrt(P / lam)) and breaks the fit; linearised as
   published it scores at chance, and norm-matched it equals isotropic
   noise, 0.07-0.17 AUROC below the projected posterior on every bio task.

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
  scored 0 (fixed in dbb42ac). Rerun on the 215 scored answers (E18):
  projected 0.561 ± 0.018, MAP entropy 0.566.
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

- **Date / job.** 2026-10-25, rerun 2026-10-26 with the Dolly fix (dbb42ac)
  and the batched token posterior (‖δ‖ 8.38); `15_baselines.lsf`; answer
  tokens. Dolly: 215 of 256 sequences have scored answer tokens.
- **Results.**

  | score | passes | WikiText | Dolly | CodeAlpaca | GSM8K | Spanish | held-out over tol |
  |---|---|---|---|---|---|---|---|
  | MAP entropy | 1 | 0.937 | 0.566 | 0.167 | 0.297 | 0.696 | - |
  | token projected (MI) | 8 | 0.867 | 0.562 | 0.496 | 0.820 | 0.819 | 1.8% |
  | token isotropic (MI) | 8 | 0.295 | 0.419 | 0.344 | 0.349 | 0.515 | 99.9% |
  | LoRA ensemble, 8 seeds (MI) | 8 | 0.276 | 0.425 | 0.439 | 0.496 | 0.575 | 63.8% |
  | DropConnect p=0.1 (MI) | 8 | 0.333 | 0.430 | 0.309 | 0.376 | 0.510 | 18.6% |
  | k-NN, layer 15 | 1 | 0.938 | 0.750 | 0.888 | 0.895 | 0.957 | - |
  | Mahalanobis, layer 15 | 1 | 0.914 | 0.671 | 0.895 | 0.876 | 0.910 | - |
  | k-NN / Mahalanobis, layer 30 | 1 | 0.83 / 0.65 | 0.73 / 0.64 | 0.40 / 0.75 | 0.18 / 0.25 | 0.92 / 0.95 | - |

  Error AUROC (test + near-OOD): MAP 1-conf 0.865; all sampling methods raw
  0.77-0.79, within entropy deciles 0.53-0.56; hidden-state distances 0.40-0.50.
  Ensemble members are ‖δ‖ = 22.1 apart; DropConnect 5.2.
- **Showed.** The projected posterior is the best of all sampling-based
  methods, including the deep-ensemble reference (whose members disagree most
  in-distribution). But one-pass mid-layer k-NN is the better input-novelty
  detector on every set. The posterior's case must rest on what input distance
  cannot see: unknown facts on normal-looking inputs (E17).

## E15 Scaling curve: constrained sequences 64 -> 1024

- **Date / job.** 2026-10-25/26; `16_scaling.lsf`; same 976 held-out sequences
  (train[1024:2000]) for every n; answer tokens; at own alpha and rescaled to
  ‖δ‖ = 8.25.
- **Results** (matched norm 8.25):

  | n constrained | 64 | 128 | 256 | 512 | 1024 |
  |---|---|---|---|---|---|
  | held-out over tol | 16.4% | 5.4% | 1.8% | **0.1%** | **0.1%** |
  | held-out over tol, own alpha | 31.9% | 9.9% | 1.8% | 1.1% | 0.1% |
  | held-out mean abs dNLL | 0.0038 | 0.0030 | 0.0025 | 0.0020 | 0.0015 |
  | rank (constraint directions) | ~4,600 | ~8,200 | ~14,700 | ~30,000 | ~60,900 |
  | rank per constrained token | 72% | 64% | 57% | ~58% | 57% |
  | WikiText / GSM8K / Spanish AUROC | 0.81 / 0.74 / 0.79 | 0.84 / 0.79 / 0.81 | 0.87 / 0.82 / 0.82 | 0.88 / 0.86 / 0.84 | 0.88 / 0.87 / 0.86 |
  | projection time (H100) / residual | 0.8 h / 0.005 | 1.6 h / 0.010 | 2.9 h / 0.018 | 5.9 h / 0.045 | 11.8 h / 0.093 |

- **Showed.** Held-out violations fall at least ~3x per doubling and reach
  the resolution floor (1 of 976 sequences; worst change 0.012 vs tol 0.0114)
  from 512 sequences on. OOD detection improves and flattens (WikiText 0.88
  at 512 and 1024). The rank per constrained token settles at ~57%: beyond
  256 sequences new data adds new directions at a constant rate, not a
  falling one. Cost grows linearly (11.8 h at 1024) and, at fixed 500
  sweeps, so does the residual (0.093 at 1024): large constraint sets need
  more sweeps, or the top-k approximation (E16).

## E16 Top-k directions vs the exact kernel

- **Date / job.** 2026-10-26; `17_topk.lsf` (fe77c9e). Top 2048 right
  singular vectors of token-mode J^L (256 sequences) by randomized subspace
  iteration (3 iterations + Rayleigh-Ritz, 4.4 h on H100); samples remove the
  top k on the exact posterior's noise; answer tokens; held-out =
  train[256:2000].
- **Results.** Energy share of ||J^L||_F^2 in the top k: 16 -> **64%**,
  64 -> 74%, 256 -> 84%, 1024 -> 93%, 2048 -> **96%** (of ~14.7k constraint
  directions). Distance from the kernel ||J delta|| / ||delta||: random 2.8,
  k=2048 0.45, exact 0.018.

  | method | ‖δ‖ own alpha | held-out over tol, own alpha / at ‖δ‖ 8.25 | WikiText | Dolly | CodeAlpaca | GSM8K | Spanish |
  |---|---|---|---|---|---|---|---|
  | exact projected | 8.38 | 1.8% / 1.7% | 0.867 | 0.562 | 0.496 | 0.820 | 0.819 |
  | top-2048 | 6.99 | 8.8% / 21.9% | 0.837 | 0.544 | 0.497 | 0.793 | 0.823 |
  | top-1024 | 5.53 | 6.5% / 45.9% | 0.821 | 0.533 | 0.473 | 0.754 | 0.809 |
  | top-256 | 3.83 | 3.2% / 75.1% | 0.792 | 0.514 | 0.431 | 0.656 | 0.783 |
  | top-64 | 2.53 | 0.9% / 93.9% | 0.677 | 0.488 | 0.386 | 0.547 | 0.719 |
  | top-16 | 1.33 | 0.1% / 98.4% | 0.487 | 0.450 | 0.347 | 0.426 | 0.645 |
  | isotropic | 8.38 | 99.9% / 99.8% | 0.295 | 0.419 | 0.344 | 0.349 | 0.515 |

  (OOD AUROCs at own alpha; at matched norm they differ by < 0.03.)
  Error AUROC unchanged across all (raw 0.77-0.79, within deciles 0.53-0.55).
- **Showed.** J^L is strongly concentrated. The **OOD signal lives in the top
  directions**: 2048 directions (14% of the rank) recover 94-100% of the
  exact posterior's AUROCs. But **the fit guarantee needs the tail**: at the
  same sample size top-2048 still breaks the tolerance on 22% of unseen
  sequences (exact 1.7%); with alpha measured for it, it stays safe on the
  constrained data by shrinking the samples (‖δ‖ 6.99), at 8.8% held-out.
  A top-k posterior is a workable approximation with a dial: more k, more
  safety at a given size.

## E18 Answer-token error bars over posterior seeds

- **Date / job.** 2026-10-26; `19_answer_only_seeds.lsf` (H100, 27 min).
  E11's answer-token evaluation repeated on the token-mode posteriors of
  seeds 0-2 (same theta_map, fresh projection noise). Mean ± std over seeds.
- **Results** (answer tokens; AUROC of per-sequence MI, MAP: entropy):

  | | MAP entropy | token projected | token isotropic |
  |---|---|---|---|
  | WikiText | 0.937 | 0.861 ± 0.011 | 0.336 ± 0.036 |
  | Dolly (215 scored) | 0.566 | 0.561 ± 0.018 | 0.410 ± 0.011 |
  | CodeAlpaca | 0.167 | 0.504 ± 0.012 | 0.333 ± 0.014 |
  | GSM8K | 0.297 | **0.834 ± 0.014** | 0.352 ± 0.016 |
  | Spanish Alpaca | 0.696 | **0.815 ± 0.013** | 0.549 ± 0.033 |

  | | projected | isotropic |
  |---|---|---|
  | ‖δ‖ | 8.83 ± 0.40 | 8.97 ± 0.41 |
  | test NLL (MAP 1.7764) | 1.7763 ± 0.0001 | 1.7761 ± 0.0004 |
  | held-out train over tol | 3.5% ± 1.6% | 100% |
  | error AUROC, MI raw (MAP 1-conf 0.854) | 0.778 ± 0.001 | 0.768 ± 0.004 |
  | error AUROC, MI within entropy deciles | 0.556 ± 0.001 | 0.554 ± 0.002 |
  | logistic error AUROC, softmax -> softmax+MI | 0.854 -> 0.853 | 0.854 -> 0.853 |

- **Showed.** E11 holds up: seed-to-seed spread is ~0.01-0.02 AUROC, far
  below the gaps that matter (GSM8K +0.54 and Spanish +0.12 over entropy,
  +0.3-0.5 over isotropic). Dolly carries no signal for any method once the
  truncated answers are dropped: its answers look like Alpaca's. The error
  result is equally stable and equally negative: MI adds nothing to the
  softmax for wrong tokens, projected and isotropic alike.

## E19 Projection residual: 500 vs 1500 sweeps

- **Date / job.** 2026-10-26; H100 pipeline with `N_ITER=1500`
  (`QUEUE=gpuh100 SKIP_FINETUNE=1 SKIP_SEQUENCE=1 N_ITER=1500`), outputs
  tagged `_it1500`. Same theta_map, seed and noise as the 500-sweep token
  posterior; all-token evaluation and tolerance sweep.
- **Results.**

  | | 500 sweeps | 1500 sweeps |
  |---|---|---|
  | kernel residual mean / max | 0.0155 / 0.0183 | **0.0052 / 0.0072** |
  | projection time (H100) | 2.9 h | 8.0 h |
  | measured alpha / ‖δ‖ | 6,347 / 8.38 | 6,256 / 8.44 |
  | worst train dNLL (tol 0.0114) | 0.01142 | 0.01143 |
  | test NLL (MAP 1.3248) | 1.3249 | 1.3249 |
  | WikiText AUROC (all tokens) | 0.991 | 0.990 |
  | train / test MI | 3.11e-4 / 3.87e-4 | 3.17e-4 / 3.93e-4 |

  Tolerance sweep (tol 0.5%-20% of MAP NLL): alpha 0-5% lower at 1500 sweeps
  (samples 0-3% larger), worst train dNLL, test NLL and WikiText AUROC equal
  to within 0.003 except at tol 5% (0.700 -> 0.662, where the curve is steep).
- **Showed.** At 256 constrained sequences 500 sweeps are enough: tripling
  them cuts the residual 3x and changes no result. The residual left over at
  500 sweeps only cost ~2% of sample size (the measured alpha absorbs it).
  Does not settle the larger runs: at n = 1024 the residual was 0.093 (E15),
  5x this one, so those may still want more sweeps.

## E20 k-NN vs posterior MI: fair data, complementarity

- **Date / job.** 2026-10-26; `20_knn_vs_mi.lsf` (7e6ae93, H100, 33 min).
  `baselines_single_pass.py` with the token posterior's (seed 0) per-sequence
  MI; answer tokens; hidden-state distances fitted on all 2000 fine-tuning
  sequences (208k tokens, k-NN bank 50k of them, as E14) or on the
  projection's 256 only (25.7k tokens).
- **Results.** k-NN layer 15, bank from 2000 / 256 sequences: WikiText
  0.938 / 0.933, Dolly 0.750 / 0.750, CodeAlpaca 0.888 / 0.872, GSM8K
  0.895 / 0.906, Spanish 0.957 / 0.954 (Mahalanobis similar, within 0.02).

  MI vs k-NN layer 15 (bank from 2000 sequences):

  | set | MI alone | k-NN alone | MI within k-NN quintiles | z(log MI) + z(k-NN) |
  |---|---|---|---|---|
  | WikiText | 0.867 | 0.938 | **0.814** | **0.949** |
  | Dolly | 0.562 | 0.750 | 0.519 | 0.675 |
  | CodeAlpaca | 0.496 | 0.888 | 0.473 | 0.818 |
  | GSM8K | 0.819 | 0.895 | **0.663** | 0.898 |
  | Spanish | 0.819 | 0.957 | **0.758** | **0.965** |

  (bank from 256: MI within quintiles 0.79 / 0.53 / 0.49 / 0.71 / 0.76.)
  Per-sequence Spearman rho(MI, k-NN) within a set 0.06-0.22 (CodeAlpaca
  -0.17), rho(MI, MAP entropy) 0.26-0.73, rho(k-NN, entropy) -0.29-0.23.
  Against the last-layer distances MI is the stronger score on GSM8K and
  WikiText (k-NN L30 0.18 / 0.83, MI within its quintiles 0.68 / 0.90).
- **Showed.** k-NN's win is not a data advantage: 8x less bank data costs it
  <= 0.02. But the two scores measure different things: they are nearly
  uncorrelated per sequence, and MI keeps most of its separation among
  sequences equally far from training on the three sets where it works.
  An unweighted sum improves on k-NN there (+0.01) and hurts where MI is at
  chance (Dolly -0.08, CodeAlpaca -0.07), so the combination is not a free
  win. Correction to an earlier guess: within a set MI tracks the MAP's
  entropy (and the gradient-curvature product), not input distance.

## E17 Knowledge coverage: synthetic biographies

- **Date / job.** 2026-10-26/27; `18_bio.lsf` (H100, 13.3 h; projection
  11.9 h). 240 fictional people x 4 facts, f = 0/1/2/4/8/16 training
  biographies per person (40 people each; never-seen names recombined from
  seen first and last names); 1,240 documents, 59k tokens. LoRA r = 8 on q/v
  + MLP (P = 1.98M), 20 epochs. Token-mode posterior on the whole corpus:
  68k J^L rows, 8 samples x 1000 sweeps, residual 2.6e-3; measured alpha at
  1% of the MAP NLL (0.105 -> tol 0.00105 nats) gives ‖δ‖ = 2.97. Baselines:
  isotropic at the same norm, 8-member LoRA ensemble (seeds 1-8, members
  ‖δ‖ ~81 from theta_map), DropConnect p = 0.1 (‖δ‖ 19). 960 cloze probes,
  teacher-forced on the true answer, mean over answer tokens.
- **MAP precondition.** Exact-match accuracy f = 0..16: 3.7 / 85.6 / 92.5 /
  96.9 / 99.4 / 99.4%; confidence 86% on never-seen people (**confident
  hallucination**); mean answer entropy 0.45 -> 0.02.
- **Results** (AUROC raw / within MAP-entropy quintiles; Spearman with
  rarity -log2(1+f)):

  | score | unseen (f=0 vs >=1) | rare (f<=1 vs >=8) | wrong (42 vs 758) | Spearman |
  |---|---|---|---|---|
  | MAP entropy | 0.859 / 0.598 | 0.900 / 0.659 | 0.918 / 0.708 | 0.628 |
  | MAP 1-conf | 0.865 / 0.637 | 0.904 / 0.669 | 0.939 / **0.822** | 0.638 |
  | **projected MI** | 0.907 / **0.861** | 0.934 / **0.804** | 0.941 / 0.793 | 0.698 |
  | isotropic MI | 0.879 / 0.706 | 0.904 / 0.615 | 0.930 / 0.767 | 0.648 |
  | LoRA ensemble MI | **0.975 / 0.946** | **0.974 / 0.955** | 0.861 / 0.697 | **0.810** |
  | DropConnect MI | 0.874 / 0.636 | 0.912 / 0.666 | 0.905 / 0.655 | 0.650 |
  | k-NN / Mahalanobis L15 | 0.51 / 0.39, 0.48 / 0.44 | 0.50 / 0.35, 0.47 / 0.38 | 0.52 / 0.34, 0.54 / 0.45 | ~0 |
  | k-NN L30 | 0.602 / 0.513 | 0.777 / 0.725 | 0.586 / 0.453 | 0.440 |

  Mean projected MI by f: 1.4e-3 / 4.6e-4 / 2.1e-4 / 1.1e-4 / 4.5e-5 /
  1.3e-5 (monotone, 100x from f = 0 to 16).
- **Showed.** The first task where the posterior clearly earns its keep:
  where input distance is blind (mid-layer k-NN at chance by construction)
  and the softmax is overconfident, projected MI tracks how often a fact was
  seen, well beyond the MAP's entropy (+0.26 / +0.15 over MAP 1-conf within
  quintiles) and beyond equal-size isotropic noise (+0.16 / +0.19), so the
  projection's structure, not noise sensitivity, carries the signal. It
  does not beat the softmax on the MAP's own errors (0.79 vs 0.82). The deep
  ensemble is stronger on coverage (0.95), at 8 full fine-tunes and samples
  27x farther from theta_map; it is weaker on errors (0.70).
- **Caveats.** Posterior samples are small (tol = 0.001 nats, since the MAP
  NLL is 0.105): the size-matched comparison with the ensemble is open.
  Within-quintile AUROC conditions on entropy only; a logistic softmax + MI
  test needs per-probe scores (not saved). "wrong" has 42 positives.

## E21 Bio posterior size sweep, fit cost, logistic test

- **Date / job.** 2026-10-27; `21_bio_size.lsf` (8ccd65e, H100). The E17
  samples rescaled from the measured ‖δ‖ 2.97 to 6 / 12 / 25 / 50 / 81 (81 =
  the ensemble members' distance from theta_map); fit cost = per-document
  mean NLL change on the corpus vs theta_map (MAP 0.105, tol 0.00105);
  held-out logistic regression, 2 folds split by person, softmax = MAP
  entropy + 1-conf, plus log MI.
- **Results.** Within-entropy-quintile AUROC (unseen / rare / wrong) and fit:

  | method | ‖δ‖ | unseen | rare | wrong | corpus mean dNLL | worst | docs over tol |
  |---|---|---|---|---|---|---|---|
  | projected | 2.97 | 0.861 | 0.804 | 0.793 | 0.0000 | 0.001 | 0% |
  | projected | 6 | 0.857 | 0.802 | 0.790 | 0.0001 | 0.004 | 4% |
  | projected | 12 | 0.847 | 0.800 | 0.789 | 0.0005 | 0.017 | 38% |
  | projected | 25 | 0.788 | 0.792 | 0.764 | 0.004 | 0.073 | 88% |
  | projected | 50 | 0.602 | 0.623 | 0.443 | 0.137 | 0.475 | 100% |
  | projected | 81 | 0.426 | 0.353 | 0.363 | 0.752 | 1.731 | 100% |
  | isotropic | 2.98 | 0.706 | 0.615 | 0.767 | 0.0001 | 0.038 | 43% |
  | isotropic | 25 | 0.601 | 0.586 | 0.568 | 0.028 | 0.550 | 97% |
  | isotropic | 81 | 0.396 | 0.335 | 0.353 | 1.195 | 2.814 | 100% |
  | LoRA ensemble | 80.9 | **0.946** | **0.955** | 0.697 | 0.0001 | 0.267 | 97% |
  | DropConnect | 19.1 | 0.636 | 0.666 | 0.655 | 0.004 | 0.517 | 89% |

  Held-out logistic AUROC, softmax -> + log MI: unseen 0.864 -> **0.909**
  (projected) / 0.878 (isotropic) / **0.975** (ensemble); rare 0.900 ->
  0.933 / 0.906 / **0.976**; wrong 0.935 -> **0.952** / 0.946 / 0.932.
  Projected flat up to ‖δ‖ 12-25 (0.909 -> 0.902), down at 50-81 (0.84-0.85).
- **Showed.** The ensemble's lead is **not a sample-size effect**. The
  posterior's ranking is scale-free while it stays linear (‖δ‖ 3-12), then
  collapses once the samples leave the region where the linearisation holds:
  at the ensemble's distance the projected samples raise the corpus NLL by
  0.75 nats (7x the MAP's), while the ensemble members, just as far away,
  fit it as well as theta_map on average (mean dNLL 0.0001). The ensemble
  samples the *curved* set of good solutions; the projected posterior only
  its tangent at theta_map. Within that tangent, MI is a real addition to
  the softmax (held-out +0.045 / +0.033 on coverage, and for the first time
  on errors, +0.017, 42 positives) and beats equal-size noise throughout.
- **Next.** Follow the curved set: take a kernel step, then pull back onto
  the zero-loss-change set with a few Gauss-Newton or short fine-tune
  corrections, and re-linearise (a retraction). That would give samples at
  ensemble-like distances that still keep the fit.

---

## E22 Knowledge coverage at equal inference cost

- **Date / job.** 2026-10-27; `22_bio_equal_cost.lsf` (20f5300, H100, 5 min).
  E17's 8 ensemble members and 8 projected samples split into disjoint
  groups of K = 2 (4 groups) and K = 4 (2 groups); mean ± std over groups.
- **Results.** Within-entropy-quintile AUROC / held-out logistic (softmax +
  log MI):

  | method | K | unseen | rare | wrong |
  |---|---|---|---|---|
  | LoRA ensemble | 2 | 0.908 ± 0.031 / 0.943 | 0.901 ± 0.013 / 0.952 | 0.682 ± 0.038 / 0.928 |
  | LoRA ensemble | 4 | 0.935 ± 0.001 / 0.963 | 0.936 ± 0.003 / 0.968 | 0.696 ± 0.056 / 0.931 |
  | LoRA ensemble | 8 | 0.946 / 0.975 | 0.955 / 0.976 | 0.697 / 0.932 |
  | projected | 2 | 0.746 ± 0.033 / 0.885 | 0.738 ± 0.030 / 0.917 | 0.670 ± 0.122 / 0.936 |
  | projected | 4 | 0.829 ± 0.010 / 0.902 | 0.787 ± 0.023 / 0.929 | 0.769 ± 0.049 / 0.949 |
  | projected | 8 | 0.861 / 0.909 | 0.804 / 0.933 | 0.793 / 0.952 |

  (logistic std over groups <= 0.02.)
- **Showed.** On coverage the ensemble wins at every equal inference cost:
  a 2-member ensemble beats the full 8-sample posterior. The posterior's MI
  also needs more samples to settle (2 -> 8: unseen 0.75 -> 0.86; ensemble
  0.91 -> 0.95). On errors the posterior keeps its small edge at K = 4-8
  (0.77-0.79 vs 0.70 within; logistic 0.95 vs 0.93), noisy at K = 2. The
  matching only inference cost flatters the posterior: in training compute
  it is far more expensive. E17's job ran 13.3 h; the projection alone took
  45,259 s (12.6 h) for 8 samples x 1000 sweeps, while the MAP fine-tune
  and the 8 ensemble fine-tunes plus the MAP probe check fit in the job's
  first ~30 min (all 9 adapters existed by 08:59 for an 08:29 start), i.e.
  ~3 min per fine-tune. One projection costs ~200-250 fine-tunes of GPU time
  on the same H100. At equal training compute the ensemble dominates on
  coverage; the posterior's remaining arguments are keeping the fit (it is
  the same model on the training data, the members are not: 97% of
  documents over tol) and its edge on errors.

## E23 Curved (retracted) posterior

- **Date / job.** 2026-10-27; `23_bio_retract.lsf` (20f5300, H100; retraction
  5.3 h for 8 samples). Each E17 projected direction u walked out in steps
  of <= 4, each followed by 10 L-BFGS iterations on mean_t (l_t - l_t^MAP)^2
  over the corpus tokens; saved at arc length 6 / 12 / 25 / 50 / 81.
- **Results.** Fit (mean / worst per-document dNLL, docs over tol) and
  within-entropy AUROC, retracted vs the linear rescaling of E21 at the same
  ‖δ‖:

  | ‖δ‖ | fit retracted | fit linear (E21) | unseen ret / lin | rare ret / lin | wrong ret / lin |
  |---|---|---|---|---|---|
  | 2.97 | (exact) 0.0000 / 0.001 / 0% | | 0.861 | 0.804 | 0.793 |
  | 6 | 0.0000 / 0.001 / 0% | 0.0001 / 0.004 / 4% | 0.860 / 0.857 | 0.803 / 0.802 | 0.788 / 0.790 |
  | 12 | 0.0002 / 0.002 / 2% | 0.0005 / 0.017 / 38% | 0.855 / 0.847 | 0.801 / 0.800 | 0.775 / 0.789 |
  | 25 | 0.0010 / 0.010 / 54% | 0.004 / 0.073 / 88% | 0.816 / 0.788 | 0.801 / 0.792 | 0.754 / 0.764 |
  | 50 | 0.0089 / 0.062 / 94% | 0.137 / 0.475 / 100% | 0.790 / 0.602 | 0.799 / 0.623 | 0.665 / 0.443 |
  | 81 | 0.0548 / 0.212 / 100% | 0.752 / 1.731 / 100% | 0.734 / 0.426 | 0.815 / 0.353 | 0.458 / 0.363 |
  | ensemble 81 | 0.0001 / 0.267 / 97% | | 0.946 | 0.955 | 0.697 |

  Held-out logistic unseen: 0.909 (<= 25) -> 0.901 (50) -> 0.871 (81).
  cos(δ, u) = 1.000 at every arc length: the corrections are tiny next to ‖δ‖.
- **Showed.** The pull-back works as a fit repair: it cuts the corpus NLL
  increase 14x at ‖δ‖ 81 (0.75 -> 0.055 nats) and removes E21's collapse
  (unseen 0.43 -> 0.73, rare 0.35 -> 0.82). But it never beats the small
  posterior: the signal is best at ‖δ‖ 3-12 and decays with distance, and
  stays far below the ensemble at the ensemble's distance. Two differences
  remain. The retracted samples still fit worse on average (+0.055 nats, a
  +52% bias on the MAP NLL) while the ensemble members are unbiased (mean
  0.0001, with per-document changes of both signs, 97% over tol). And the
  retracted samples point along u, a kernel direction at theta_map, while
  the ensemble's directions come from independent training runs. So the
  E21 reading "the posterior only sees the tangent" is incomplete: following
  the curve with this corrector does not recover the ensemble's diversity.
  What makes the ensemble disagree on unseen facts is where its members
  land, not how far they are from theta_map.

## E24 Top-k posterior on the biographies, noise seeds

- **Date / job.** 2026-10-27; `24_bio_topk.lsf` (20f5300, H100, 5.3 h;
  subspace 3.5 h). Top-1024 right singular directions of the bio J^L (68k
  rows, 3 subspace iterations), samples restricted to the complement of the
  top k, measured alpha; noise seeds 0 (E17's noise), 1, 2.
- **Results.** Energy share 0.51 / 0.73 / 0.93 at k = 64 / 256 / 1024; the
  measured alpha gives ‖δ‖ 0.35-1.35, all docs within tol. Within-entropy
  AUROC (seed 0 / 1 / 2):

  | method | unseen | rare | wrong | logistic unseen |
  |---|---|---|---|---|
  | exact projected | 0.861 | 0.804 | 0.793 | 0.909 |
  | top-64 | 0.709 / 0.718 / 0.663 | 0.653 / 0.603 / 0.623 | 0.727 / 0.749 / 0.747 | 0.874-0.878 |
  | top-256 | 0.714 / 0.707 / 0.689 | 0.701 / 0.646 / 0.646 | 0.706 / 0.744 / 0.745 | 0.876-0.880 |
  | top-1024 | 0.727 / 0.707 / 0.698 | 0.722 / 0.695 / 0.705 | 0.717 / 0.772 / 0.746 | 0.879-0.881 |
  | isotropic | 0.706 | 0.615 | 0.767 | 0.878 |

- **Showed.** Unlike Alpaca OOD (E16, top-2048 kept 94-100%), top-k keeps
  little of the coverage signal: at k = 1024 (93% of the energy) unseen is
  at the isotropic level and rare recovers only ~half the exact posterior's
  gain over isotropic. The facts seen once or twice give small-gradient
  constraints in the spectrum's tail; a top-k posterior leaves exactly those
  free, so rarely-seen facts look like unseen ones. Input novelty lives in
  the top directions, knowledge coverage in the tail, so this use needs
  the full projection (or a much larger k). Seed spread: within-quintile
  AUROC moves by up to 0.05 between noise draws at 8 samples (logistic
  <= 0.006), the same size as E22's group std; differences below ~0.05
  within quintiles are within noise.

---

## E25 Cost curve: posterior quality against training GPU time

- **Date / job.** 2026-10-27/28; `25_bio_cost_alt.lsf`, `26_bio_cost_pcg.lsf`
  (523a606, H100, 4.9 h / 5.3 h). E17's bio posterior recomputed with
  snapshots after n iterations, each with its own measured alpha: alternating
  projections (E17's method, which ran 1000 sweeps) at 25 / 50 / 100 / 250,
  and block-Jacobi preconditioned CG (new, `project_vectors_pcg`; one
  iteration = the passes of one sweep) at 10 / 25 / 50 / 100 / 250. One more
  fine-tune timed on the same GPU: **166 s** training (185 s with loading).
- **Cost of a posterior** = per-batch (J_b J_b^T)^+ precompute (38 min, once)
  + sweeps (41 s each for 8 samples) + measured alpha (13 min). Ensemble =
  K x 166 s. (Both also need the MAP fine-tune; an ensemble can count it as
  a member.)
- **Results** (within-entropy-quintile AUROC; all posteriors within tol on
  every document):

  | method | GPU min | ≈ fine-tunes | residual | ‖δ‖ | unseen | rare | wrong |
  |---|---|---|---|---|---|---|---|
  | ensemble K = 2 | 5.5 | 2 | | 81 | 0.908 | 0.901 | 0.682 |
  | ensemble K = 3 | 8.3 | 3 | | 81 | 0.919 | 0.926 | 0.725 |
  | ensemble K = 4 | 11 | 4 | | 81 | 0.935 | 0.936 | 0.696 |
  | ensemble K = 8 | 22 | 8 | | 81 | 0.946 | 0.955 | 0.697 |
  | alternating 25 | 68 | 25 | 0.023 | 2.95 | 0.859 | 0.803 | 0.788 |
  | alternating 50 | 85 | 31 | 0.013 | 2.90 | 0.861 | 0.801 | 0.798 |
  | alternating 100 | 119 | 43 | 0.0074 | 2.96 | 0.861 | 0.801 | 0.791 |
  | alternating 250 | 223 | 81 | 0.0043 | 2.99 | 0.861 | 0.802 | 0.791 |
  | alternating 1000 (E17) | 767 | 277 | 0.0026 | 2.97 | 0.861 | 0.804 | 0.793 |
  | PCG 10 | 63 | 23 | 1.13 | 0.19 | 0.789 | 0.645 | 0.795 |
  | PCG 25 | 74 | 27 | 0.46 | 0.38 | 0.842 | 0.793 | 0.794 |
  | PCG 100 | 127 | 46 | 0.092 | 1.82 | 0.860 | 0.802 | 0.786 |
  | PCG 250 | 235 | 85 | 0.063 | 2.49 | 0.859 | 0.804 | 0.792 |
  | isotropic | | | | 2.99 | 0.706 | 0.615 | 0.767 |

- **Showed.** (1) E17 over-solved by 40x: 25 sweeps give the full signal
  and fit (the measured alpha absorbs the larger residual at the same ‖δ‖),
  so the posterior costs ~1.1 h instead of 12.8 h, and the precompute is now
  more than half of it. (2) Still ~25 fine-tunes: on coverage a 2-member
  ensemble (2 fine-tunes) beats it at any posterior budget. (3) On wrong
  answers the posterior beats every ensemble size (0.79 vs 0.68-0.73) at
  every snapshot. (4) **PCG is a negative result**: block-Jacobi CG
  converges more slowly than accelerated alternating projections (residual
  0.46 vs 0.023 at 25 iterations). The 78 batches share their dominant
  gradient directions, which an additive (Jacobi) preconditioner counts 78
  times over; the multiplicative sweep (block Gauss-Seidel) does not. A
  Krylov method would need the sweep itself as preconditioner (symmetric
  Gauss-Seidel), not block Jacobi. Default stays alternating.
- **Next.** The remaining cost is the precompute (34k rows of J J^T at one
  VJP + JVP per row over a batch of 16 sequences: cost ∝ rows x batch size,
  so a smaller --proj-batch-size cuts it) and alpha. Even at ~0 projection
  cost, the coverage ranking would not flip; what the posterior offers at
  any price here is the error signal and keeping the fit.

---

## E25b Is the posterior's edge on wrong answers real? (bootstrap)

- **Date / job.** 2026-10-28; local, on E25's saved per-probe scores
  (`bio_eval_cost_alt.json`), 1000 paired bootstrap resamples of the
  "wrong" task (42 wrong / 758 right), within-entropy AUROC.
- **Results.** Posterior (25 sweeps) 0.788, 8-member ensemble 0.697, MAP
  1-conf 0.822, isotropic 0.767. Posterior minus ensemble +0.095, 95% CI
  [-0.030, +0.245] (P <= 0: 0.06); minus MAP 1-conf -0.027 [-0.140, +0.078];
  minus isotropic +0.026 [-0.079, +0.134].
- **Showed.** With 42 errors the edge over the ensemble is suggestive, not
  established, and the posterior does not beat MAP 1-conf at all within
  entropy quintiles. Point 7's "the posterior keeps an edge on wrong
  answers" needs a test with many more errors (proposed E28).

## E26 Posterior on a pretrained model: PopQA / TriviaQA

- **Date / job.** 2026-10-28; `27_pretrained_qa.lsf` (78bdee7, H100, ~5 h).
  SmolLM2-360M as pretrained; zero-initialised LoRA (r 8, q/v/gate/up/down),
  posterior on lora_B (P = 1.88M), token-mode on 512 FineWeb-Edu documents x
  128 tokens (65k rows), batch 2, 50 sweeps (projection incl. precompute
  2.4 h); measured alpha -> ‖δ‖ 79. Closed-book few-shot QA, greedy answer;
  2000 PopQA (accuracy 10.2%), 2000 TriviaQA (32.4%). Scores on the model's
  own answer. LoRA ensemble: 4 fine-tunes (2 epochs) on the same 512 docs.
- **Fit.** Constraint docs: 0% over tol (worst 0.027 = tol); held-out
  FineWeb docs: **6.7%** over tol (isotropic 59%, ensemble 64%). With 65k
  token constraints the fit generalises to unseen text of the same kind.
- **Results** (AUROC raw / within MAP-entropy quintiles):

  | score | PopQA rare vs popular | PopQA wrong | TriviaQA wrong |
  |---|---|---|---|
  | MAP entropy | 0.675 / 0.571 | 0.744 / 0.488 | 0.794 / 0.594 |
  | MAP NLL of the answer | 0.671 / 0.576 | 0.777 / 0.664 | 0.816 / 0.675 |
  | **semantic entropy** (10 samples) | 0.570 / 0.508 | **0.867 / 0.779** | **0.850 / 0.756** |
  | 1 - P(True) | 0.445 / 0.467 | 0.549 / 0.561 | 0.614 / 0.642 |
  | projected MI | 0.455 / 0.363 | 0.717 / 0.654 | 0.683 / 0.482 |
  | isotropic MI | 0.470 / 0.340 | 0.687 / 0.590 | 0.668 / 0.426 |
  | LoRA ensemble MI | 0.511 / 0.390 | 0.700 / 0.577 | 0.685 / 0.474 |
  | k-NN L16 | 0.443 / 0.504 | 0.622 / 0.663 | 0.483 / 0.605 |

  Spearman with -log popularity: projected MI -0.05 (entropy +0.24).
  Logistic softmax -> + score: popqa_wrong 0.746 -> 0.753 (MI), 0.762 (SE),
  0.858 (NLL); trivia_wrong 0.791 -> 0.786 (MI), 0.805 (SE), 0.841 (NLL).
  25 vs 50 sweeps: identical.
- **Showed.** A negative result with a clear reason. The posterior keeps
  the fit, even on held-out text, but its MI does not see fact popularity
  (below chance) and adds nothing on errors beyond the softmax; semantic
  entropy and the answer's own NLL are far better. MI measures how tightly
  the *constraint data* pins a prediction: a 512-document sample of generic
  text contains essentially none of the PopQA facts, so all facts are equally
  unconstrained. In the biographies (E17) the constraint data were the facts.
  The posterior can only know what its constraints contain, so the
  "posterior on a pretrained model" needs constraints that cover the
  pretraining data's facts (at their frequencies) -- i.e. projection on a
  representative share of the pretraining corpus, and E24 says the coverage
  signal lives in the spectrum's tail, so top-k will not do.

## E27 Speed-ups, part 1: TF32 and projection batch size

- **Date / job.** 2026-10-28; `28_bio_speedups.lsf` elements 1-2 (78bdee7).
- **Results.** H1 TF32 (batch 16): precompute 30.9 min vs 38.3 fp32
  (1.24x), sweeps 38 s vs 41 s (1.08x), fine-tune 158 s vs 166 s (1.05x); and
  the alpha search **failed**: under TF32 even the smallest grid step
  changed a document's NLL by more than tol (0.00105 nats), i.e. TF32's
  rounding noise in the per-token losses exceeds the tolerance. H2 batch 4
  (+TF32): precompute 27.6 min (1.4x vs fp32 batch 16, not the ~4x a FLOP
  count predicts). Elements 2-3 killed (TF32 alpha would fail the same way).
- **Showed.** H1 refuted: the passes are not matmul-bound, and TF32 is too
  noisy for the alpha measurement (fixed: alpha and residuals now always
  fp32). H2 mostly refuted: smaller batches do less work per row but run
  less efficiently. Both point to overhead-bound passes -> H5 (job 29).

## E27 Speed-ups, part 2: fewer sweeps, faster alpha search

- **Date / job.** 2026-10-28; `28_bio_speedups.lsf` element 4 (fp32, batch
  16, row_batch 16; snapshots at 5 / 10 / 25 / 50 sweeps, each with its own
  measured alpha; minimum-first alpha search).
- **Results.** Precompute 39 min, sweeps 41 s.

  | sweeps | sweep time | residual | alpha | ‖δ‖ | unseen | rare | wrong | worst dNLL |
  |---|---|---|---|---|---|---|---|---|
  | 5 | 3.5 min | 0.114 | 1.33e6 | 1.22 | 0.854 | 0.788 | 0.787 | 0.0011 |
  | 10 | 6.9 min | 0.056 | 3.26e5 | 2.46 | 0.856 | 0.800 | 0.801 | 0.0012 |
  | 25 | 17 min | 0.023 | 2.26e5 | 2.95 | 0.859 | 0.803 | 0.788 | 0.0010 |
  | 50 | 35 min | 0.013 | 2.36e5 | 2.89 | 0.861 | 0.801 | 0.798 | 0.0010 |

  (MI AUROC within MAP-entropy quintiles; isotropic 0.705 / 0.615 / 0.768.
  Held-out logistic unseen 0.908-0.909 and rare 0.930-0.932 at every sweep
  count. No sample over tol at any snapshot; isotropic 41%.)
  H4: alpha at 25 sweeps 2.2635e5 vs E25's full search 2.2597e5 (0.2%),
  search time 3.7 min vs 13 min (3.5x).
- **Showed.** H3 holds: 5 sweeps keep the signal within the seed spread
  (E24: up to 0.05), 10 sweeps match 25-50. The less converged samples keep
  more row-space content, so the measured alpha shrinks them (‖δ‖ 1.22 at
  5 sweeps) and the fit is safe at any sweep count; the ranking survives
  the smaller scale. H4 holds. With these, the sweeps and alpha are minor
  and the precompute dominates: batch 16 at 10 sweeps ~39 + 7 + 4 = 50 min
  (~18 fine-tunes, vs 68 min in E25 and 12.6 h in E17). Batch 4 / rb 256
  could give ~8 + 26 + 4 = 38 min (~14 fine-tunes) if its sweeps converge
  as fast per sweep (untested: smaller batches = more, smaller blocks).
  Bio-specific: on the pretrained run (E26) the residual converges ~2x
  slower per sweep (0.067 at 25), so a stopping rule (alpha change between
  snapshots < a few %) is safer than a fixed count. Even at ~18 fine-tunes
  the coverage conclusion of E25 stands (the 2-member ensemble beats it).

## E27 Speed-ups, part 3: where the precompute time goes

- **Date / job.** 2026-10-28; `29_bench_precompute.lsf` (902f374, H100,
  fp32). One batch's J J^T timed per row_batch (rows per vmapped VJP/JVP),
  plus a sweep of 8 vectors, extrapolated to the full bio run (25 sweeps).
- **Results** (full-run estimates; E25 measured precompute 38 min, 25
  sweeps 17 min at batch 16 / row_batch 16):

  | batch | precompute rb 16 | rb 64 | rb 128 | rb 256 | 25 sweeps |
  |---|---|---|---|---|---|
  | 16 | 38 min | 34 min (1.15x, 40 GB) | OOM | OOM | 17 min |
  | 4 | 27 min | 9.6 min | 9.0 min | 8.4 min (3.4x, 30 GB) | 64 min |
  | 1 | 29 min | 11 min | 11 min | 11 min (3.1x) | 253 min |

- **Showed.** H5 holds where each call does little work: at batch 1-4,
  64+ rows per call cut the precompute ~3x (overhead-bound), at batch 16
  memory caps it at 1.15x (compute-bound). But the sweeps go the other way:
  small batches make them overhead-bound (4x / 15x slower at batch 4 / 1),
  and alternating projections are sequential over batches, so they cannot
  be merged. Best totals at 25 sweeps: batch 16 / rb 64 ~51 min vs 55 now;
  batch 4 / rb 256 ~72 min. **The setup is near its floor on this
  hardware**; the remaining lever is the number of sweeps (H3: if 10 suffice,
  batch 4 / rb 256 gives ~34 min, ~12 fine-tunes).

## E28 Error detection with ~500 errors, three replications

- **Date / job.** 2026-10-28; `30_bio_errors.lsf` array 1-3 (42550f4, H100).
  Each element its own corpus seed (people, facts, theta_map, posterior,
  8-member ensemble); 40 / 300 / 300 / 100 / 40 / 40 people at f = 0 / 1 /
  2 / 4 / 8 / 16 (2260 biographies, 3280 probes). Posterior: batch 16, 25
  sweeps (snapshot at 10), minimum-first alpha. Paired bootstrap with
  `bootstrap_bio.py` (`hpc_results/bootstrap_bio_err.json`).
- **Results.** MAP accuracy f = 1 / 2 / 4: 0.67-0.73 / 0.86-0.91 /
  0.96-0.99; 584 / 561 / 443 wrong answers. Posterior ‖δ‖ 2.5-2.7, no
  document over tol (isotropic 38-41%; ensemble ‖δ‖ 111, 96-97%).
  Within-entropy AUROC per replication:

  | task | posterior (25) | ensemble (8) | MAP 1-conf | isotropic |
  |---|---|---|---|---|
  | unseen | 0.774 / 0.814 / 0.800 | 0.871 / 0.896 / 0.869 | 0.580 / 0.595 / 0.613 | 0.654 / 0.637 / 0.695 |
  | rare | 0.845 / 0.862 / 0.787 | 0.977 / 0.980 / 0.979 | 0.745 / 0.659 / 0.781 | 0.706 / 0.540 / 0.707 |
  | wrong | 0.823 / 0.758 / 0.801 | 0.732 / 0.777 / 0.801 | 0.723 / 0.702 / 0.716 | 0.735 / 0.649 / 0.727 |

  Posterior minus (mean over replications, 95% CI from probes resampled
  within each): wrong vs ensemble +0.024 [+0.009, +0.061], vs MAP 1-conf
  +0.081 [+0.061, +0.108], vs isotropic +0.091 [+0.068, +0.114]; unseen vs
  ensemble -0.082 [-0.110, -0.057]; rare vs ensemble -0.147 [-0.192,
  -0.112]. 10 sweeps = 25 sweeps on every task (gaps <= 0.003).
  Held-out logistic on wrong, softmax -> + posterior MI / + ensemble MI:
  0.848 -> 0.878 / 0.847; 0.867 -> 0.882 / 0.872; 0.885 -> 0.906 / 0.892.
  Cost: precompute 71 min, 25 sweeps 31 min, alpha 6.7 min = 109 min,
  ~21 fine-tunes (311 s each); the 8-member ensemble 41 min.
- **Showed.** With ~10x the errors the picture is clear on two points and
  mixed on one. (1) The posterior beats the softmax on wrong answers in
  every replication (+0.08 within quintiles, +0.015-0.030 logistic), so
  E25b's "no better than MAP 1-conf" was a small-sample artefact. (2) The
  ensemble wins coverage by a wide margin everywhere. (3) Against the
  ensemble on wrong answers the posterior is ahead on average, but by
  replication +0.091 / -0.019 / 0.000: the CI counts probe noise with the
  three worlds fixed, so the honest reading is "at least as good, ahead in
  one of three". Its clearer edge is complementarity: the ensemble's MI
  adds ~0.00-0.007 to the softmax on errors, the posterior's 0.015-0.030
  in all three. At ~2.5x the ensemble's training compute.

## E28b Ensemble + posterior combined

- **Date / job.** 2026-10-28; local (`combine_bio.py`, 0 GPU) on E28's
  per-probe scores (`hpc_results/combine_bio_err.json`). Held-out logistic
  on softmax (entropy + 1-conf) + log MI of posterior (25 sweeps), ensemble
  (8), or both; 2 folds by person, mean over 10 random person splits. Plus
  an unfitted score: mean of the two MIs' ranks, within entropy quintiles.
- **Results** (replications 1 / 2 / 3, mean):

  | task | softmax | +post | +ens | +post+ens | rank mean (within) |
  |---|---|---|---|---|---|
  | unseen | 0.786 | 0.851 | 0.904 | **0.921** (0.926 / 0.928 / 0.908) | 0.902 (ens alone 0.879) |
  | rare | 0.909 | 0.937 | 0.988 | **0.990** (0.990 / 0.992 / 0.988) | 0.971 (ens 0.979) |
  | wrong | 0.868 | 0.889 | 0.871 | **0.898** (0.884 / 0.892 / 0.917) | 0.817 (post 0.794, ens 0.770) |

- **Showed.** The two are complementary: both MIs together beat either
  alone in every replication on unseen (+0.017 over the ensemble) and wrong
  (+0.009 over the posterior, +0.027 over the ensemble); rare is at the
  ceiling. The unfitted rank mean shows the same without any fitting
  (unseen 0.902 vs 0.879, wrong 0.817 vs 0.794 / 0.770). Caveat: the
  combination has 16 forward passes and ~29 fine-tunes of training compute
  against 8 / 8 for the ensemble; a 16-member ensemble is the missing
  control for "more members would do the same".

## E28c 16-member ensemble control

- **Date / job.** 2026-10-30; `31_bio_ens16.lsf` array 1-3 (091c975, H100):
  8 more fine-tunes (seeds 9-16, ~300 s each) per E28 replication, evaluated
  with the posterior; `combine_bio.py` locally
  (`hpc_results/combine_bio_err_ens16.json`). ens8 = E28's members 1-8,
  ens8b = the new 9-16, ens16 = all (one MI); ens8+ens8b = both MIs as two
  features.
- **Results** (held-out logistic AUROC, mean of 3 replications; 16 forward
  passes marked *):

  | task | softmax | +post | +ens8 | +ens16* | +ens8+ens8b* | +post+ens8* | +post+ens16 |
  |---|---|---|---|---|---|---|---|
  | unseen | 0.786 | 0.851 | 0.904 | 0.915 | **0.922** | 0.921 | 0.927 |
  | rare | 0.909 | 0.937 | 0.988 | 0.992 | **0.994** | 0.990 | 0.993 |
  | wrong | 0.868 | 0.889 | 0.871 | 0.873 | 0.873 | **0.898** | 0.899 |

  Wrong per replication, post+ens8 vs ens16: 0.884 / 0.892 / 0.916 vs
  0.850 / 0.873 / 0.894. Within entropy quintiles (unfitted rank mean):
  wrong post+ens8 0.817 vs ens16 0.777, ens8+ens8b 0.786; unseen 0.902 vs
  0.893 / 0.903; rare 0.971 vs 0.986 / 0.989.
- **Showed.** At equal inference cost (16 passes) the posterior's gain on
  wrong answers is not a member count effect: 8 more members add 0.002 to
  the ensemble on errors, the posterior 0.025 (+0.03 / +0.02 / +0.02 by
  replication). On coverage 8 more members do as well as or better than
  the posterior (unseen tie, rare +0.004). So the posterior adds something
  the ensemble lacks only on errors; on coverage a bigger ensemble is the
  better buy (and cheaper: 16 fine-tunes vs 8 + ~21 for post+ens8).

---

## E30 Why the posterior catches errors (bio, E28's replications)

- **Date / job.** 2026-10-31; `33_bio_error_breakdown.lsf` array 1-3 (a772320,
  gpuh100, ~50 s per element), `error_breakdown.py` on E28's
  `bio_eval_err{1,2,3}_ens16.json`: 3120 seen probes per replication,
  584 / 561 / 443 wrong. Within-entropy AUROC (5 strata of MAP entropy),
  mean of the three replications.
- **Results.** By error type (n = errors per replication, averaged):

  | error type | n | post | ens16 | ens8 | 1-conf | train fit |
  |---|---|---|---|---|---|---|
  | swap (another training value) | 130 | **0.780** | 0.708 | 0.710 | 0.714 | 0.660 |
  | other | 400 | **0.803** | 0.790 | 0.780 | 0.720 | 0.704 |
  | first answer token wrong | 281 | 0.767 | **0.842** | 0.833 | 0.666 | 0.667 |
  | later answer tokens wrong | 248 | **0.792** | 0.701 | 0.695 | 0.709 | 0.691 |

  By training frequency (rep 3 has too few f = 4 errors to report):

  | f | n_wrong | post | ens16 | ens8 | 1-conf | train fit |
  |---|---|---|---|---|---|---|
  | 1 | 369 | **0.807** | 0.728 | 0.723 | 0.777 | 0.696 |
  | 2 | 147 | **0.875** | 0.732 | 0.717 | 0.872 | 0.726 |
  | 4 | 16 | 0.925 | 0.688 | 0.691 | **0.934** | 0.775 |

  Held-out logistic (folds by name), mean of 3: softmax 0.868, +train fit
  0.857, **+post 0.889**, +train fit + post **0.898**, +ens16 0.873,
  +train fit + ens16 0.890, +ens8 0.871, +train fit + ens8 0.887.
  Spearman with the fact's training-fit NLL: post 0.547, ens16 0.269,
  1-conf 0.641. Overlap at the 20% most uncertain seen probes, post vs
  ens16: both 205, post only 123, ens16 only 79, neither 122 (per
  replication); post vs train fit: post only 179, train fit only 102.
- **Showed.** The mechanism is **partiality of the fit, not the fit
  itself**. The posterior's edge over the softmax is largest exactly where
  the fact was seen often enough to be fit (f = 2: 0.875 vs 0.872; f = 4:
  0.925 vs 0.934, i.e. a tie) but it still beats the softmax by +0.03 at
  f = 1, and it beats the 16-member ensemble by +0.14 / +0.14 / +0.24 at
  f = 1 / 2 / 4. The ensemble inverts this: it is *best* on errors it can
  only catch as novelty (first-token errors, 0.842 vs the posterior's
  0.767) and near-chance on later-token errors (0.701), where the first
  token is right and only the continuation is wrong. The posterior catches
  the opposite class: a near-miss continuation (`Oslo` -> ` L`, `Lille`
  -> ` Bille`) moves a token the constraints had seen, so the samples
  disagree about it, whereas every ensemble member makes the same near-miss
  and the ensemble cannot see it. The posterior also tracks the training
  fit far more closely than the ensemble does (rho 0.55 vs 0.27) but still
  adds beyond it in the logistic test (0.889 -> 0.898), and flags 179
  errors the training fit misses. So the gain is not "the ensemble has not
  converged": it is that a projected posterior retains a local sensitivity
  to individual constrained tokens that independent fine-tunes average
  away.

## E29 PopQA knowledge injection (three replications)

- **Date / job.** 2026-10-31 to 11-02; `32_popqa_errors.lsf` array 1-3
  (element 2) and `pb_popqa_fix[1,3]` (elements 1, 3; null-subject fix).
  2450 probes per replication; theta_map, 16-member LoRA ensemble,
  posterior (25 sweeps). `combine_bio.py` (`hpc_results/combine_popqa.json`)
  and `bootstrap_bio.py` (`hpc_results/bootstrap_popqa.json`) over all three.
- **Results.** MAP accuracy f = 0 / 1 / 2 / 4 / 8: 0.11-0.13 / 0.92-0.93 /
  0.98 / 0.99-1.0 / 1.0, so only 76 / 82 / 82 wrong answers (bio: ~550).
  Within-entropy AUROC per replication:

  | task | post | ens16 | ens8 | MAP 1-conf | isotropic |
  |---|---|---|---|---|---|
  | unseen | 0.621 / 0.638 / 0.773 | 0.707 / 0.851 / 0.770 | 0.722 / 0.866 / 0.779 | 0.716 / 0.653 / 0.730 | 0.518 / 0.443 / 0.595 |
  | rare | 0.789 / 0.635 / 0.825 | 0.932 / 0.905 / 0.916 | 0.929 / 0.890 / 0.924 | 0.789 / 0.585 / 0.799 | 0.686 / 0.390 / 0.649 |
  | wrong | 0.834 / 0.703 / 0.837 | 0.772 / 0.791 / 0.784 | 0.758 / 0.800 / 0.761 | **0.881 / 0.816 / 0.885** | 0.724 / 0.712 / 0.825 |

  Posterior minus (bootstrap): wrong vs ens16 +0.009 [-0.059, +0.073], vs
  MAP 1-conf -0.069 [-0.162, -0.001]; unseen vs ens16 -0.098 [-0.161,
  -0.042]; rare vs ens16 -0.168; vs isotropic +0.16 / +0.18 on coverage.
  Held-out logistic (mean of 3): wrong softmax 0.940, + post 0.943, + ens16
  0.940, + post + ens8 0.944, + ens8 + ens8b 0.940; unseen 0.921 / 0.923 /
  0.949 / 0.952 / 0.949; rare 0.917 / 0.922 / 0.962 / 0.960 / 0.964.
  Error types (`error_breakdown_popqa*.json`): 61-69 of the ~80 errors per
  replication have the first answer token wrong; swaps 7-10, later-token
  errors 13-20.
- **Showed.** E28's error-detection result does not carry over to real
  facts in this design. The softmax is already at 0.94 on errors and
  neither the posterior (+0.003) nor the ensemble (+0.000) adds to it; the
  MAP's own confidence beats both within entropy strata. On coverage the
  ensemble wins as on bio, the posterior still beats isotropic but adds
  nothing to the softmax. Consistent with E30's mechanism: the posterior's
  edge on bio is on near-miss continuations (swaps, later-token errors),
  and PopQA's few errors are mostly first-token misses, the class where it
  is weakest. Confounds: the base model already knows many PopQA facts
  (11-13% right at f = 0), so the fine-tune mostly re-teaches and leaves
  ~80 errors per replication; a harder injection (less training, or facts
  the base model cannot know) would be needed to test real facts with as
  many errors as bio.
## E31 Full fine-tuning vs LoRA (bio, three replications)

- **Date / job.** 2026-11-02 to 11-03; `34_bio_full.lsf` elements 1-3
  (job 29872208). E28's three bio corpora; theta_map by full fine-tuning
  (all 135M weights, lr 2e-5, 20 epochs), an 8-member full ensemble, and
  the projected posterior over all weights (10 sweeps) with an isotropic
  prior (`post`) and a per-tensor RMS-scaled prior (`post_rms`). Compared
  with E28/E28c's LoRA world on the same corpora, each world on its own
  MAP's errors (`bio_full_eval{1,2,3}.json`,
  `error_breakdown_bio_full{1,2,3}.json`, `bootstrap_bio_full.json`,
  `logs/29872208_*_full.out`; figure `plot_full_vs_lora.py`).
- **Results.** The full MAP answers fewer probes: f = 1 / 2 / 4 accuracy
  0.41-0.46 / 0.67-0.70 / 0.84-0.90 (LoRA 0.67-0.73 / 0.86-0.90 / 0.96),
  so 1181 / 1053 / 1140 errors (LoRA 584 / 561 / 443). Within-entropy
  AUROC, rep 1 / 2 / 3:

  | task | post | post_rms | isotropic | full ens8 | MAP 1-conf | LoRA post | LoRA ens8 |
  |---|---|---|---|---|---|---|---|
  | unseen | 0.666 / 0.698 / 0.703 | 0.666 / 0.697 / 0.688 | 0.617 / 0.628 / 0.588 | **0.761 / 0.766 / 0.731** | 0.554 / 0.612 / 0.522 | 0.774 / 0.814 / 0.800 | 0.871 / 0.896 / 0.869 |
  | rare | 0.691 / 0.635 / 0.636 | 0.681 / 0.627 / 0.638 | 0.581 / 0.603 / 0.547 | **0.876 / 0.682 / 0.843** | 0.665 / 0.655 / 0.573 | 0.845 / 0.862 / 0.787 | 0.977 / 0.980 / 0.979 |
  | wrong | 0.753 / 0.742 / 0.717 | 0.754 / 0.740 / 0.719 | 0.686 / 0.684 / 0.668 | **0.767 / 0.754 / 0.730** | 0.697 / 0.676 / 0.694 | 0.823 / 0.758 / 0.801 | 0.732 / 0.777 / 0.801 |

  Paired bootstrap, full post minus: on wrong, full ens8 -0.013 [-0.027,
  -0.000], MAP 1-conf +0.048 [+0.037, +0.062], post_rms -0.000 [-0.003,
  +0.003], isotropic +0.058; on unseen, ens8 -0.064 [-0.096, -0.032],
  1-conf +0.126; on rare, ens8 -0.146. Held-out logistic on wrong, gain
  over the softmax (0.812 / 0.816 / 0.811): post +0.023 / +0.021 / +0.009,
  post_rms the same, isotropic +0.003 / +0.007 / 0.000, full ens8 +0.034 /
  +0.026 / +0.017 (LoRA world: post +0.030 / +0.015 / +0.021, ens8 -0.001
  / +0.005 / +0.007). Error types (within entropy, mean of 3): later-token
  errors post 0.835 vs ens 0.803; swaps 0.835 vs 0.827; first-token errors
  0.637 vs 0.737; other values 0.720 vs 0.740. Fit: both projected
  posteriors keep every training sequence within tolerance (worst dNLL
  0.0011), isotropic breaks it on 29-34%, the full ensemble (‖δ‖ 11.6-11.8
  vs 1.9-2.5) on 95%. post and post_rms flag the same top-20% errors
  (91-93% overlap).
  Cost (H100): a full fine-tune takes 242 s (LoRA 311 s), the projection
  of 8 samples over 134.5M weights 5982-7045 s (iso) and 6576-6691 s (rms)
  for 122k J^L rows, i.e. ~100-110 min per prior, ~2x LoRA's 10-sweep
  posterior (~50 min, E27) and ~25 full fine-tunes; the 8-member full
  ensemble costs 32 min. A whole element (MAP, two posteriors, ensemble,
  evaluation) ran 4.4-4.5 h. At 10 sweeps the kernel residual is
  0.10-0.14 (iso) and 0.02-0.03 (rms), far above LoRA's, but the measured
  alpha absorbs it: the fit holds (above). Measured alpha 3.5e7 / 3.0e7 /
  2.1e7 (iso, ‖δ‖ 1.97 / 2.12 / 2.51); rms 2.3e6 / 9.2e5 / 9.9e5 in
  z = θ/s (‖δ‖ in θ 1.36 / 2.16 / 2.09), RMS per tensor 0.04-1.82.
- **Showed.** (1) The RMS prior changes nothing: same AUROCs (bootstrap
  gap 0.000 on wrong), same flagged errors. E31's hypothesis that an
  isotropic prior over all tensors is badly scaled, and that LoRA works
  mainly as a sensible prior, is not supported; the projection, not the
  prior scale, sets what moves. (2) The posterior still beats the softmax
  on errors (+0.048, CI excludes 0) and adds to it in the logistic test,
  and its edge is again on later-token errors, with the ensemble ahead on
  first-token errors (E30's split reproduces in all three). (3) But in the
  full world the ensemble is the better error detector, narrowly (-0.013,
  CI just touching 0) and with a larger logistic gain in every
  replication, whereas LoRA's ensemble added ~nothing to the softmax:
  full fine-tunes from different seeds diverge more than LoRA runs and
  so disagree on errors too. (4) Everything is weaker on coverage than
  with LoRA (posterior unseen -0.11, ensemble -0.12), plausibly because
  the full MAP has not learnt many of the f = 1 / 2 facts it is scored as
  having seen. (5) At ~2x LoRA's posterior cost and ~3x the full
  ensemble's, the full-weight posterior is not worth it here. Confound:
  the full MAP's accuracy is well below LoRA's, so the two worlds' error
  sets differ in size and kind (below).
- **Why the full MAP is less accurate (2026-11-03).** Not underfitting:
  its corpus NLL is lower than LoRA's (0.110 / 0.109 / 0.107 vs 0.130 /
  0.127 / 0.121),
  and on the facts it gets wrong the answer tokens in the training
  documents are fit better (median NLL 0.003-0.009 vs 0.01-0.04, f = 2 / 4,
  from `error_breakdown` examples). It recites the documents but cannot
  answer the probe. Split by whether the probe sentence ("X studied at Y")
  occurs verbatim in training (corpora regenerated with
  `make_bio_corpus.py`), f = 1 accuracy rep 1 / 2 / 3: LoRA 0.76 / 0.74 /
  0.77 when it does, 0.63 / 0.65 / 0.70 when only another phrasing does;
  full 0.51 / 0.59 / 0.55 and 0.36 / 0.39 / 0.38. Both lose on paraphrase, but the full model fails even
  on the exact sentence, without the rest of the biography before it.
  **Corrected by E35:** the explanation offered here (full fine-tuning
  binds facts to their document context, LoRA stores them retrievably) is
  wrong. It was the learning rate: full fine-tuning at lr 1e-4 answers more
  probes than LoRA, also in phrasings never seen for that fact.

## E35 Full fine-tuning: learning rate and phrasings

- **Date / job.** 2026-11-05; `35_full_lr_sweep.lsf` (job 29900277, 36 min).
  `probe_phrasings.py`: every fact asked in all three phrasings,
  teacher-forced exact match, "seen" = that exact sentence is in training;
  "all" = the fact answered in every phrasing. E28's LoRA and E31's full
  MAPs on corpora 1-3, plus full fine-tuning on corpus 1 at other
  settings (`hpc_results/phrasings_*.json`).
- **Results.** f = 1 (corpus 1; LoRA / E31 full on corpora 2, 3 within
  ~0.05 of these):

  | model | probe | seen | unseen | any | all |
  |---|---|---|---|---|---|
  | LoRA, lr 1e-3, 20 ep | 0.672 | 0.687 | 0.565 | 0.787 | 0.373 |
  | full, lr 2e-5, 20 ep (E31) | 0.409 | 0.528 | 0.330 | 0.660 | 0.162 |
  | full, lr 2e-5, 40 ep | 0.487 | 0.575 | 0.390 | 0.708 | 0.200 |
  | full, lr 5e-5, 20 ep | 0.606 | 0.612 | 0.500 | 0.749 | 0.288 |
  | full, lr 1e-4, 20 ep | 0.760 | 0.759 | 0.666 | 0.862 | 0.467 |
  | full, lr 1e-4, 40 ep | **0.789** | **0.784** | 0.704 | **0.882** | 0.513 |
  | full, lr 3e-4, 20 ep | 0.738 | 0.771 | 0.693 | 0.849 | **0.562** |

  At f = 2 the same order (full lr 1e-4 0.933, 40 ep 0.949, LoRA 0.855);
  f = 0 stays at 0.03-0.06 everywhere. Final training loss does not track
  any of this (0.104-0.123 for all settings: most of it is the
  unpredictable choice of template and person).
- **Showed.** E31's full MAP was undertrained, not limited by full
  fine-tuning: from lr 1e-4 it beats LoRA on every column, including facts
  asked in a phrasing never seen for them (0.67-0.70 vs 0.57), and lr 3e-4
  gives the most facts known in every phrasing (0.56 vs 0.37). E31's
  full-vs-LoRA comparison therefore ran at a poor setting and should be
  repeated at lr 1e-4 (20 epochs: same cost as E31's MAP). Both kinds of
  model still lose ~0.1 on rephrased facts.

## E36 Sketched J J^T precompute

- **Date / job.** 2026-11-05; `36_sketch.lsf` (job 29900278). `--sketch
  192`: per batch, S J_b J_b^T S^T for a Gaussian S (192 x rows) instead of
  J_b J_b^T (~760 rows), grown by 64 if the kept rank exceeds k - 16 (it
  never did). 10 sweeps, same seed (same eps) as the exact run. Element 1:
  LoRA on E28's corpus 1, exact and sketched in the same job; element 2:
  the full model on E31's corpus 1 (lr 2e-5 MAP), vs E31's exact run.
- **Results.**

  | | precompute | sweeps | projection | kept rank | alpha | ‖δ‖ |
  |---|---|---|---|---|---|---|
  | LoRA exact | 5367 s | 804 s | 6171 s | 25,513 | 3.34e5 | 2.42 |
  | LoRA sketch | **1084 s** (4.95x) | 807 s | **1890 s** (3.3x) | 22,031 | 5.30e5 | 1.93 |
  | full exact (E31) | ~5400-5900 s | ~600 s | 6025 s | | 3.46e7 | 1.97 |
  | full sketch | **1403 s** (~4x) | ~700 s | **2101 s** (2.9x) | 19,637 | 4.74e7 | 1.69 |

  Same-eps samples, exact vs sketched (LoRA): cosine 0.9992-0.9993 for all
  8; the norm ratio 0.794 is exactly sqrt(alpha_exact / alpha_sketch), i.e.
  the projections agree and only the measured scale differs. Within-entropy
  AUROC exact / sketch: LoRA unseen 0.773 / 0.775, rare 0.840 / 0.843,
  wrong 0.824 / 0.817; full 0.666 / 0.660, 0.691 / 0.679, 0.753 / 0.749.
  Logistic gains over the softmax equal to 0.001. No sequence over tol.
  (This node ran ~2x slower than E27's: LoRA's exact precompute took 89
  min vs 39.)
- **Showed.** The sketch gives the same posterior for a fifth of the
  precompute: the samples point the same way (cos 0.999), and the AUROCs
  and fit are unchanged within 0.012. It keeps ~14% fewer directions above
  the cutoff (22.0k vs 25.5k kept rank), which leaves a little more row
  space in the samples, so the measured alpha shrinks them by ~20%; the
  ranking does not care. Projection time falls 3.3x (LoRA) / 2.9x (full),
  and the sweeps are now ~43% of it. On E27's hardware this makes the LoRA
  posterior ~8 + 7 + 4 = ~19 min, ~4 LoRA fine-tunes (half the 8-member
  ensemble); 5 sweeps (E27: nearly as good) would give ~15 min. Next
  levers: fewer sweeps, and k (192 never had to grow: per-batch rank <= 176).

## E37 Full fine-tuning at lr 1e-4 vs LoRA (bio, three replications)

- **Date / job.** 2026-11-05 to 11-06; `37_bio_full_lr.lsf` (job 29901454).
  E31's design at E35's learning rate: full theta_map (lr 1e-4, 20 epochs),
  8-member full ensemble, projected posterior over all weights with the
  sketched precompute (192), 10 sweeps, isotropic prior
  (`bio_full{s}_lr1e-4_eval.json`, `error_breakdown_bio_full{s}_lr1e-4.json`,
  `phrasings_bio_full{s}_lr1e-4.json`, `bootstrap_bio_full_lr1e-4.json`).
- **Results.** The full MAP now matches or beats LoRA: f = 1 accuracy
  0.76 / 0.69 / 0.73 (LoRA 0.67 / 0.68 / 0.73), in every phrasing 0.47 /
  0.39 / 0.40 (LoRA 0.37 / 0.34 / 0.38); 378 / 550 / 433 errors (LoRA 584
  / 561 / 443). Within-entropy AUROC, mean of 3 (LoRA world in brackets):

  | task | full post | full ens8 | full ens2 | MAP 1-conf |
  |---|---|---|---|---|
  | unseen | 0.754 (0.796) | **0.864** (0.879) | 0.839 | 0.650 (0.596) |
  | rare | 0.789 (0.831) | **0.920** (0.979) | 0.901 | 0.632 (0.728) |
  | wrong | 0.710 (0.794) | **0.773** (0.770) | 0.762 | 0.611 (0.713) |

  Bootstrap, full post minus: ens8 unseen -0.110 [-0.145, -0.051], rare
  -0.131, wrong -0.063 [-0.099, +0.027]; MAP 1-conf on wrong +0.099
  [+0.043, +0.120]. Held-out logistic on wrong: the softmax alone is
  already 0.913, + post +0.008, + ens8 +0.002 (LoRA world: softmax 0.867,
  + post +0.022). The posterior still leads on later-token errors in two
  of three replications (0.80-0.81 vs ens 0.58-0.83) and trails on
  first-token errors (0.55-0.66 vs 0.70-0.87). Fit: posterior ‖δ‖
  2.5-6.2, 0% of sequences over tol; ensemble members ‖δ‖ 63-78 (E31:
  11.6), 96% over.
- **Showed.** With a well-trained full MAP the full-weight posterior is
  worse than the LoRA one on every task (-0.04 to -0.08), and the full
  ensemble beats it everywhere (wrong: -0.063, CI touching 0). LoRA's
  error-detection edge (E28) does not carry over to full fine-tuning: the
  full MAP's softmax already ranks its errors well in the logistic test
  (0.913) and neither score adds much. The posterior still beats the
  MAP's confidence within entropy strata and keeps the fit. For this
  paper, LoRA is the setting where the projected posterior is worth it.

## E38 Sketched LoRA posterior vs LoRA ensembles at equal cost

- **Date / job.** 2026-11-05 to 11-06; `38_lora_equal_cost.lsf` (job
  29901456). E28's corpora and theta_maps; posterior with `--sketch 192`
  and the VJP-only Gram, 10 sweeps with a 5-sweep snapshot (each its own
  measured alpha); E28's members 1-8 as 2-, 4- and 8-member ensembles
  (`bio_sketch_eval{s}.json`, `posterior_bio_sketch{s}.json`,
  `bootstrap_bio_sketch.json`).
- **Cost.** Precompute 476-479 s (E36's JVP sketch: 1084 s on a slower
  node; E27's exact: 39 min); 5 sweeps 374-381 s, 10 sweeps 748-762 s;
  alpha search 286-334 s for both snapshots. Posterior at 5 sweeps
  ~1000-1100 s, at 10 ~1400-1500 s (the alpha time covers both
  snapshots, so a single run's is about half). A LoRA fine-tune on the
  same node took 302 / 310 / 314 s (seed 99, logs 29901456_*), so the
  posterior costs ~3.3 fine-tunes at 5 sweeps and ~4.5 at 10.
- **Results.** Within-entropy AUROC, mean of 3 (ens2 / ens4 = mean over
  the disjoint member groups):

  | task | post 5 sweeps | post 10 | ens2 | ens4 | ens8 | MAP 1-conf |
  |---|---|---|---|---|---|---|
  | unseen | 0.788 | 0.790 | 0.824 | 0.861 | 0.879 | 0.596 |
  | rare | 0.814 | 0.823 | 0.931 | 0.962 | 0.979 | 0.728 |
  | wrong | **0.784** | **0.786** | 0.736 | 0.759 | 0.770 | 0.713 |

  Bootstrap, post (5 sweeps) minus: on wrong, ens2 [2:0] +0.037 [+0.018,
  +0.069], ens4 [4:0] +0.015 [-0.001, +0.053], ens8 +0.014 [-0.003,
  +0.051], MAP 1-conf +0.070; on unseen, ens2 -0.054 [-0.080, -0.025];
  on rare, ens2 -0.112. Held-out logistic gain over the softmax on wrong:
  post +0.021, ens2 -0.006, ens4 0.000, ens8 +0.004; on unseen post
  +0.060 vs ens2 +0.081, ens4 +0.107.
- **Showed.** At ~3.3 fine-tunes the LoRA posterior is the best of the
  four scores on wrong answers: clearly ahead of a 2-member ensemble,
  narrowly ahead of 4 and 8 members (CIs touching 0), and the only one
  that adds to the softmax in the logistic test. On coverage even 2
  members beat it. 5 sweeps = 10 sweeps. The posterior's niche is
  catching errors on facts the model was trained on, at a cost between a
  3- and a 4-member ensemble.

## E39 SmolLM2-360M: posterior vs ensembles at equal cost

- **Date / job.** 2026-11-06; `39_bio_360m.lsf` (job 29905055). E28's bio
  corpora 1-3 on SmolLM2-360M (LoRA P = 3.52M), E38's design: sketch 192
  with the VJP-only Gram (`--max-sketch-gb 40`), 5 and 10 sweeps, members
  1-8 (`bio360_eval{s}.json`, `posterior_bio360_{s}.json`,
  `bootstrap_bio360.json`, `error_breakdown_bio360_{s}.json`).
- **Cost.** Fine-tune 362-370 s (10 per element; 135M: ~305 s).
  Precompute 927-952 s (135M: ~478 s); per-batch rank 18.8-19.8k of
  122k rows (15-16%, 27.3k J J^T columns formed); 5 sweeps 504-513 s;
  alpha 306-550 s for both snapshots. Posterior at 5 sweeps ~1600 s =
  ~4.4 fine-tunes (135M: 3.3), at 10 ~2250 s = ~6.2. The posterior's
  cost grows with P, the fine-tune's barely.
- **Accuracy.** MAP exact match at f = 1: 0.74-0.78 (135M: 0.67-0.73);
  331-404 wrong of 3120 seen probes.
- **Results.** Within-entropy AUROC, mean of 3 (ens2 / ens4 = mean over
  the disjoint member groups):

  | task | post 5 | post 10 | ens2 | ens4 | ens8 | MAP 1-conf |
  |---|---|---|---|---|---|---|
  | unseen | 0.817 | 0.818 | 0.871 | 0.898 | 0.918 | 0.558 |
  | rare | 0.801 | 0.811 | 0.912 | 0.950 | 0.974 | 0.591 |
  | wrong | 0.764 | 0.768 | 0.786 | 0.814 | 0.836 | 0.658 |

  Bootstrap, post 5 minus (first groups): on wrong, ens2 [2:0] -0.033
  [-0.077, +0.024], ens4 [4:0] -0.065 [-0.112, -0.003], ens8 -0.072
  [-0.122, -0.013], MAP 1-conf +0.105 [+0.073, +0.142]; on unseen ens2
  -0.056, on rare ens2 -0.119. Per replication on wrong the posterior is
  0.710 / 0.822 / 0.759 (ens8 0.841 / 0.815 / 0.852): large spread.
  Held-out logistic gain over the softmax on wrong: post +0.020, ens2
  -0.005, ens4 0.000, ens8 +0.004 (as at 135M); on unseen post +0.056 vs
  ens2 +0.067, ens4 +0.085.
- **Showed.** E38's headline does not carry to 360M: ensembles of 4+
  members beat the posterior on wrong answers, at a cost ratio that also
  moved against it (4.4 fine-tunes). What survives at both sizes is the
  logistic test: the posterior is the only score that adds to the softmax
  on wrong answers (+0.02); ensemble MI is mostly redundant with the
  softmax there. The per-batch rank stays a small fraction of the rows.

## E40 Harder PopQA: real facts with ~600 errors

- **Date / job.** 2026-11-06; `40_popqa_hard.lsf` (job 29905056). E29's
  PopQA corpora; MAP-only evaluations at 2 / 4 / 6 epochs gave seen
  accuracy 0.22 / 0.42-0.46 / 0.69-0.73, so 6 epochs was used (the
  largest <= 0.80). Then E38's design (`popqah_eval{s}.json`,
  `popqah{s}_ep*_map.json`, `posterior_popqah{s}.json`,
  `bootstrap_popqah.json`, `error_breakdown_popqah{s}.json`).
- **Cost.** 6-epoch fine-tune 63-68 s. Precompute 756-862 s; per-batch
  rank 33.2-33.7k of 101k rows (33%, the highest yet: an underfit model
  leaves most token gradients non-zero); 5 sweeps 280-291 s. Posterior at
  5 sweeps ~1200 s = ~18 fine-tunes: when fine-tuning is short, the
  precompute dominates.
- **Accuracy.** MAP exact match f = 1 0.45-0.49, f = 2 0.76-0.84, f = 4
  0.97-0.99; f = 0 0.11-0.14; 549-629 wrong of 2050 seen.
- **Results.** Within-entropy AUROC, mean of 3:

  | task | post 5 | post 10 | ens2 | ens4 | ens8 | MAP 1-conf |
  |---|---|---|---|---|---|---|
  | unseen | **0.704** | 0.708 | 0.650 | 0.681 | 0.689 | 0.667 |
  | rare | 0.608 | 0.607 | 0.698 | 0.724 | 0.755 | 0.627 |
  | wrong | 0.607 | 0.604 | 0.662 | 0.683 | 0.690 | 0.685 |

  Bootstrap, post 5 minus: on unseen ens2 [2:0] +0.041 [+0.010, +0.070],
  ens4 +0.018 [-0.011, +0.046], ens8 +0.014 [-0.013, +0.040], MAP 1-conf
  +0.037; on wrong ens2 -0.053 [-0.094, -0.009], ens8 -0.084, MAP 1-conf
  -0.078 [-0.123, -0.038]. Logistic gain on unseen: post +0.030, ens8
  +0.017, ens2 +0.010; on wrong all within +-0.004.
- **Showed.** On real facts with many errors the posterior does not
  detect errors (below the MAP's own confidence); it is the best unseen
  detector (ahead of 2 members, level with 4-8). The bio error-detection
  edge (E38) does not generalise to this setting. Note that these errors
  are underfitting errors (6 epochs), unlike bio's errors after 20.

## E41 Sketch size ablation

- **Date / job.** 2026-11-06; `41_sketch_ablation.lsf` (job 29905057).
  E28's LoRA corpus 1, 5 sweeps, fixed sketches (`--sketch-step 0`) of
  64 / 96 / 128 / 256 rows and the adaptive 192 (`sketch_ab_k*_eval.json`).
  Per-batch rank ~150 (E36).
- **Results.** Within-entropy AUROC (one replication):

  | k | 64 | 96 | 128 | 192 (adaptive) | 256 |
  |---|---|---|---|---|---|
  | unseen | 0.695 | 0.741 | 0.759 | 0.775 | 0.778 |
  | rare | 0.791 | 0.816 | 0.832 | 0.841 | 0.833 |
  | wrong | 0.751 | 0.780 | 0.801 | 0.812 | 0.815 |

- **Showed.** Sketches below the rank lose 0.03-0.08 AUROC; from k ~
  rank + oversampling (192) the result is flat. The adaptive default is
  right; a fixed sketch must exceed the rank.

## E42 TOFU: posterior vs ensembles at equal cost

- **Date / job.** 2026-11-06; `42_tofu.lsf` (job 29905072).
  `make_tofu_corpus.py` seeds 1-3 (40 unseen authors, 60 / 40 / 20 / 10
  at f = 1 / 2 / 4 / 8, 5 trained + 5 held-out pairs each), max length
  128, 20 epochs, "correct" = >= 80% of answer tokens; E38's design
  (`tofu_eval{s}.json`, `posterior_tofu{s}.json`, `bootstrap_tofu.json`).
- **Cost.** Fine-tune 197-204 s. Precompute 802-982 s; per-batch rank
  28.5-30.1k of 118-127k rows (24%); 5 sweeps 285-316 s. Posterior at
  5 sweeps ~1250 s = ~6 fine-tunes.
- **Accuracy.** Every trained pair is answered (token share >= 0.8) at
  every f, so there are no wrong answers: the error task is empty. f = 0
  (unseen and held-out) 0.05.
- **Results.** Within-entropy AUROC, mean of 3:

  | task | post 5 | post 10 | ens2 | ens4 | ens8 | MAP 1-conf |
  |---|---|---|---|---|---|---|
  | unseen | **0.984** | 0.985 | 0.913 | 0.946 | 0.982 | 0.868 |
  | rare | 0.946 | 0.944 | 0.968 | 0.981 | 0.985 | 0.969 |

  Bootstrap, post 5 minus: on unseen ens2 [2:0] +0.173 [+0.006, +0.198]
  (that group is the weak one; over all four groups the mean gap is
  +0.07), ens4 [4:0] +0.074 [-0.012, +0.096], ens8 +0.003 [-0.018,
  +0.024]; on rare ens8 -0.039, MAP 1-conf -0.024. Logistic gains are all
  within +-0.008 (softmax already 0.99).
- **Showed.** On TOFU the posterior matches 8 members at telling unseen
  from trained questions, ahead of 2 and 4, at ~6 fine-tunes. The error
  question needs a harder setting (fewer epochs, as in E40).

## E43 Laplace-LoRA baseline (bio, three replications)

- **Date / job.** 2026-11-06; `43_laplace_lora.lsf` (job 29908836). E28's
  corpora and theta_maps; E38's sketched posterior at 5 sweeps rerun
  (same seed), members 1-8, and `build_laplace_samples.py`: KFAC over the
  300 LoRA matrices (~110k positions, Fisher with one sampled label per
  position), prior precision by the Laplace evidence. Scored through the
  network and linearised at theta_map (`--linearize`), as drawn and
  rescaled to the posterior's mean norm (`laplace_eval{s}.json`,
  `bootstrap_laplace.json`, `posterior_lap{s}.json`).
- **Prior.** The evidence picks lam = 25.1 / 25.1 / 20.0; samples have
  ||delta|| 269-303, which is sqrt(P / lam) = 281 / 315: the prior
  dominates almost every KFAC eigenvalue (a_i g_j << lam), so the
  Laplace-LoRA posterior is close to the isotropic prior. Through the
  network the draws raise the corpus NLL by 16.9 nats per sequence on
  average (100% of sequences over the tolerance); linearised, the
  model's NLL rises by 12.3. Norm-matched (||delta|| ~1.2): 5.3% of
  sequences over the tolerance (posterior 0%, isotropic 12%).
- **Results.** Within-entropy AUROC, mean of 3 (the posterior reproduces
  E38: 0.788 / 0.814 / 0.783):

  | task | post | Laplace | Laplace lin | Laplace norm | Laplace norm lin | iso | ens2 | ens8 | MAP 1-conf |
  |---|---|---|---|---|---|---|---|---|---|
  | unseen | 0.788 | 0.507 | 0.458 | 0.655 | 0.654 | 0.663 | 0.824 | 0.879 | 0.596 |
  | rare | 0.814 | 0.491 | 0.440 | 0.646 | 0.647 | 0.651 | 0.931 | 0.979 | 0.728 |
  | wrong | **0.783** | 0.512 | 0.410 | 0.713 | 0.714 | 0.702 | 0.736 | 0.770 | 0.713 |

  Bootstrap, post minus Laplace norm-matched: unseen +0.133 [+0.104,
  +0.163], rare +0.168 [+0.117, +0.212], wrong +0.070 [+0.052, +0.094];
  linearised the same to 0.001. Logistic gain over the softmax on wrong:
  post +0.021, Laplace norm +0.002 / +0.003 (lin), as drawn -0.005 /
  -0.013.
- **Showed.** The standard Bayesian-LoRA baseline carries no signal on
  these tasks: as published (evidence prior, linearised) it is at or
  below chance, and at the posterior's norm it equals isotropic noise
  (wrong 0.713 = MAP 1-conf). KFAC curvature over the LoRA weights is
  too weak against an evidence-tuned prior to shape the samples; the
  loss projection's kernel is what makes the posterior's directions
  informative. Laplace-LoRA costs one pass over the corpus (minutes),
  far less than the posterior, but buys nothing here.

---

## Pending

- **E44** (`44_tofu_hard.lsf`): TOFU with errors. 10 trained + 5 held-out
  pairs per seen author (1300 trained probes, 3000 documents, 166k
  tokens); MAP-only evaluations at 3 / 5 / 8 / 12 epochs pick the largest
  with trained-pair accuracy <= 0.85 (correct = >= 80% of answer tokens);
  then E38's design. Does the posterior catch TOFU errors, as on bio, or
  not, as on underfit PopQA (E40)?
