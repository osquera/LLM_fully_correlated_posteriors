# Sketched precompute: why it gives the same projection

This note backs the sketched precompute in `src/llmproj/projection.py`
(`_sketched_factor`, `--sketch`). It covers plan item 7: exact equality
when the sketch is larger than the batch's rank, a bound for batches that
are only numerically low-rank, and why an error can only make the sketch
under-project. Experiments E36, E41 and E47 check it.

## 1. Setup

For one projection batch, $J \in \mathbb{R}^{n\times P}$ is the token-loss
Jacobian: one row per target token, $n \approx 800$ rows, $P \approx 2\cdot10^6$
LoRA weights. The alternating-projection sampler applies, per batch,

$$
P_E = I - J^\top (JJ^\top)^+_\tau J ,
$$

where $(\cdot)^+_\tau$ is the pseudo-inverse that keeps only eigenvalues
above $\tau = \mathrm{rtol}\cdot\lambda_{\max}$ (`rtol` $=10^{-6}$). With
the SVD $J=\sum_i \sigma_i u_i v_i^\top$, $P_E$ is the orthogonal projector
onto the complement of $\mathcal{R}_\tau(J) = \mathrm{span}\{v_i : \sigma_i^2 > \tau\}$.
Write $r$ for the numerical rank $\dim \mathcal{R}_\tau(J)$.

**The sketch.** Draw $S\in\mathbb{R}^{k\times n}$ with i.i.d. $N(0,1)$
entries and set $Y = SJ \in \mathbb{R}^{k\times P}$. Each row of $Y$ is the
gradient of one random weighted sum of token losses, so it costs one VJP.
The code then:

- forms $G = YY^\top = S\,JJ^\top S^\top$, either as $(SJ)(SJ)^\top$ (the
  VJP-only Gram) or as $S\,(JJ^\top S^\top)$ with one extra JVP per row;
- eigendecomposes $G = Q\Lambda Q^\top$ and keeps the eigenvalues above
  $\tau_Y = \mathrm{rtol}\cdot\lambda_{\max}(G)$;
- stores the batch factor as eigenvectors $U = S^\top Q$ and inverse
  eigenvalues $\Lambda^+_{\tau_Y}$.

The sweeps use this factor exactly as they use the exact one.

**Lemma 1 (what the code applies).** The sweep step is
$v \mapsto v - J^\top U\Lambda^+U^\top J v$, so the sketched batch
projector is

$$
P_S \;=\; I - J^\top S^\top Q \Lambda^+_{\tau_Y} Q^\top S J
\;=\; I - Y^\top (YY^\top)^+_{\tau_Y} Y ,
$$

the orthogonal projector onto the complement of
$\mathcal{R}_{\tau_Y}(Y)$, the span of $Y$'s right singular vectors whose
squared singular values exceed $\tau_Y$.

*Proof.* Substitute $Y = SJ$. For any matrix, $Y^\top (YY^\top)^+_\tau Y$
is the orthogonal projector onto the span of its kept right singular
vectors. $\square$

The scale of $S$ does not matter: $\tau_Y$ is relative to $\lambda_{\max}(G)$,
and the projector does not change when $Y$ is rescaled.

## 2. The sketch can only under-project

**Proposition 1.** For every $S$:

1. $\mathcal{R}_{\tau_Y}(Y) \subseteq \mathrm{row}(Y) \subseteq \mathrm{row}(J)$.
   So $P_S v = v$ for every $v \in \ker J$: the sketched projector never
   removes a direction that $J$ does not constrain.
2. Over all batches $b$, alternating sweeps with the $P_{S,b}$ converge to
   the orthogonal projection onto
   $\mathcal{K}_S = \bigcap_b \mathcal{R}_{\tau_Y}(Y_b)^\perp
   \supseteq \bigcap_b \ker J_b$.

*Proof.* (1) Every row of $Y$ is a linear combination of the rows of $J$.
(2) The cyclic product of finitely many orthogonal projectors converges to
the projector onto the intersection of their ranges (von Neumann for two,
Halperin 1962 for any finite number). By (1), each range contains
$\ker J_b$. $\square$

So a sketch that is too small leaves some constrained directions in the
sample. It never adds constraints that the data do not impose: the sketched
posterior's support contains the exact one.

Two safeguards act downstream of this:

- the kernel residual is computed with the exact $J$;
- $\alpha$ is measured on the actual per-sequence NLL.

Any under-projection therefore raises the measured NLL change, and $\alpha$
shrinks the samples to compensate. E36 shows exactly this. The sketch kept
22.0k directions where the exact precompute kept 25.5k, and its measured
$\alpha$ was $1.59\times$ larger. The norm ratio $0.794 = \sqrt{\alpha_E/\alpha_S}$
accounts for all of the difference, while the same-noise samples have
cosine 0.999.

One caveat to (1). It is stated against the true null space $\ker J$, not
against the truncated $\mathcal{R}_\tau(J)^\perp$. $Y$ can keep a direction
that lies in $J$'s row space below the exact cutoff, where
$\sigma^2 \le \mathrm{rtol}\cdot\sigma_1^2$: a direction the exact factor
leaves free, but whose loss curvature is at most $10^{-6}$ of the largest.
E47 measures this as `sketch_removes_from_exact`; the smoke test gave
$\sim 10^{-4}$ of the projection's norm.

## 3. Exact equality when $k \ge \operatorname{rank} J$

**Proposition 2.** Let $\operatorname{rank} J = r \le k$, with $J = U_r\Sigma_r V_r^\top$
the thin SVD. With no cutoff, $P_S = P_E$ with probability 1.

*Proof.* $Y = (SU_r)\Sigma_r V_r^\top$. Because $U_r$ has orthonormal
columns and the Gaussian distribution is rotation invariant,
$\Omega = SU_r$ is a $k\times r$ standard Gaussian matrix, which has full
column rank $r$ almost surely when $k\ge r$. Then $\Omega\Sigma_r$ also has
full column rank, so $\mathrm{row}(Y) = \mathrm{row}(V_r^\top) = \mathrm{row}(J)$.
Lemma 1 completes the proof. $\square$

The proof uses only that $SU_r$ has full column rank. Any sketch with that
property gives the same result, for example a subsampled randomized
Hadamard transform with enough rows. Gaussian $S$ is used because it also
gives the bounds below.

**The relative cutoff (Proposition 3).** With a cutoff, the two factors can
keep different numbers of directions, because $\Omega$ is not an isometry.
For $\Omega$ with full column rank, $\sigma_{\min}(\Omega)\,\sigma_i(J) \le
\sigma_i(Y) \le \sigma_{\max}(\Omega)\,\sigma_i(J)$ for all $i$ (Horn &
Johnson, *Topics in Matrix Analysis*, Thm 3.3.16). With
$\kappa = \sigma_{\max}(\Omega)/\sigma_{\min}(\Omega)$:

$$
\frac{\lambda_i(J)}{\lambda_1(J)}\,\kappa^{-2} \;\le\;
\frac{\lambda_i(Y)}{\lambda_1(Y)} \;\le\;
\frac{\lambda_i(J)}{\lambda_1(J)}\,\kappa^{2}.
$$

So every direction with $\lambda_i/\lambda_1 > \mathrm{rtol}\,\kappa^2$ is
kept by both factors, and every direction below $\mathrm{rtol}/\kappa^2$ is
dropped by both. The two kept ranks can differ only by the number of
eigenvalues of $J$ that fall inside that window.

For a Gaussian $\Omega$, $\kappa \lesssim (\sqrt k+\sqrt r)/(\sqrt k-\sqrt r)$
with high probability (Davidson & Szarek 2001). When $J$'s spectrum has a
gap at the cutoff, Wedin's theorem makes the kept subspaces themselves
close, not just their dimensions.

Oversampling ($k-r$) narrows the window. In practice the window is
populated sparsely: token-loss spectra fall by many decades within a few
hundred directions. E47 records $\lambda_{k}/\lambda_1$ and the kept ranks
at rtol $10^{-4}$, $10^{-6}$ and $10^{-8}$ to show this.

The window also explains the observed sign of the error. $\lambda_1(Y)$ is
inflated by the largest singular value of $\Omega$, while the small
eigenvalues of $Y$ are scaled by values nearer $\sigma_{\min}(\Omega)$. The
sketch therefore keeps somewhat fewer directions than the exact factor:
22.0k against 25.5k in E36, and 86 against 88 in the smoke test. This is
the under-projection side of Proposition 1, and it is what the measured
$\alpha$ corrects.

## 4. Numerically low rank: the randomized range finder bound

In general $J$ has full rank, with a decaying spectrum. Take a target rank
$r$, oversampling $p\ge2$ and $k=r+p$, and let $Q_Y$ be an orthonormal
basis of $\mathrm{row}(Y)$, which is the range of $J^\top S^\top$. This is
exactly the randomized range finder applied to $A = J^\top$ with test matrix
$S^\top$. Halko, Martinsson & Tropp (2011), Thm 10.6, give

$$
\mathbb{E}\,\big\|J\,(I - Q_YQ_Y^\top)\big\| \;\le\;
\Big(1+\sqrt{\tfrac{r}{p-1}}\Big)\sigma_{r+1}
+ \frac{e\sqrt{k}}{p}\Big(\sum_{j>r}\sigma_j^2\Big)^{1/2}.
$$

For $p\ge4$, Cor. 10.9 gives a bound of the same form with probability at
least $1-6p^{-p}$, at the cost of larger constants.

**Consequence.** Take any $\delta$ that the sketched projection leaves free
of the batch, $\delta \perp \mathrm{row}(Y)$. Then
$J\delta = J(I-Q_YQ_Y^\top)\delta$, so

$$
\frac{\|J\delta\|}{\|\delta\|} \;\le\; C_{r,p}\,\sigma_{r+1}
\;+\; \text{tail term}.
$$

The exact truncated projection allows a violation of
$\|J\delta\|/\|\delta\| \le \sigma_{r+1}$ too, because it leaves the
directions below its cutoff free. So once $r$ covers the directions above
the cutoff, $\sigma_{r+1}^2 \le \mathrm{rtol}\cdot\sigma_1^2$, and the
sketch's extra violation is of the order the cutoff already accepts, times
a modest constant.

The tail term is what makes the bound useful only for low-rank batches.
It is small when most token gradients are near zero, as they are for a
fine-tuned model on its own training data. It stops being small when they
are not, for a pretrained model on fresh text or an undertrained one. E47
measures where each corpus falls.

## 5. The adaptive sketch size

The code starts at $k = 192$. It adds 64 rows while the sketch's kept rank
exceeds $k - 16$, and stops at $k = n$. This is an a-posteriori check, not
a guarantee. The idea: if $J$ has $r$ directions above the cutoff and
$k \ge r + 16$, then (Proposition 3) $Y$ shows about $r$ kept directions
and $16$ spare ones. If instead the kept rank of $Y$ fills the sketch, $J$
probably has more directions than the sketch can hold, so the sketch grows.

Two failure modes remain:

- a cutoff window that hides directions (Section 3);
- an unlucky draw, which has small probability by Section 4.

Both lead to under-projection (Proposition 1), and the measured $\alpha$
then shrinks the samples.

**Observed weakness (E47).** The first failure mode does occur on
corpora with a flat spectrum. On undertrained TOFU and on Alpaca, the
sketch's kept rank drops below $k-16$, so the rule stops, while
$\lambda_k(J)/\lambda_1(J)$ is still about $2\cdot10^{-5}$. At that
point many of $J$'s directions above the cutoff lie in the window of
Proposition 3. The sketch then behaves as if the cutoff were about
$10^{-5}$ rather than $10^{-6}$. It keeps about 20% fewer directions, and
its one-batch constraint residual is 3-6% of $\|Jv\|$ against 0.1-0.5%
for the exact factor. The projection's direction is still the same
(cosine $\ge 0.9997$).

On bio the spectrum has already fallen past the cutoff at $k$
($\lambda_k/\lambda_1 \approx 5\cdot10^{-7}$), so the effect is small.
A rule that also requires the Gram's own tail to pass the cutoff with a
margin for the window,
$\lambda_{\min}(G)/\lambda_{\max}(G) < \mathrm{rtol}/\hat\kappa^2$,
would close the gap at the price of a larger $k$.

## 6. Cost

The exact factor forms $JJ^\top$ with a VJP and a JVP per row: $2n$
passes. The sketch needs $k$ VJPs plus either $k$ JVPs or one $k\times k$
Gram from the stored $Y$ ($kP$ floats: 1.5 GB at $k=192$ for LoRA). So the
cost is $k/n$ of the exact precompute, or $k/(2n)$ with the VJP-only Gram.

On the bio corpus, $r \approx 150$ of $n \approx 860$ rows, which gives
$k = 192$, about a fifth of the passes. The precompute fell from 39 to
8 min (E36, E38). Where the rank fraction is larger the saving shrinks: E47 measured
the exact rank at 58% of rows on Alpaca and 74% on undertrained TOFU,
where the sketch grows to 49% and 62% of the rows.

## 7. Empirical checks

| check | experiment | result |
|---|---|---|
| same-noise samples, exact vs sketch | E36 | cosine 0.9992-0.9993; norms differ only by $\sqrt{\alpha_E/\alpha_S}$ |
| fewer kept directions with the sketch (Prop. 3) | E36 | 22.0k vs 25.5k |
| AUROCs unchanged | E36 | within 0.012 on every task |
| $k$ below the rank under-projects (Prop. 1, 4) | E41 | $k=64, 96$: AUROC −0.03 to −0.08; flat from $k=192$ |
| sketched rank per target-token row | E39, E40, E42, E44 logs | bio 18%, TOFU 20 ep 34-36%, PopQA 6 ep 38-39%, TOFU 3 ep 58% |
| exact per-batch rank (rtol $10^{-6}$) | E47 | bio 24%, Alpaca 58%, TOFU 3 ep 74%; pretrained FineWeb pending |
| sketch never removes much of the exact projection (Prop. 1) | E47 | removed $\le 0.3\%$ of its norm; cosine $\ge 0.9997$ |
| cutoff window: fewer kept directions (Prop. 3) | E47 | 14-20% fewer than exact |
| flat spectra fool the stopping rule (Sec. 5) | E47 | TOFU 3 ep, Alpaca: one-batch residual 30-50x the exact factor's |

## References

- N. Halko, P.-G. Martinsson, J. A. Tropp. Finding structure with
  randomness. *SIAM Review* 53(2), 2011. Thm 10.6, Cor. 10.9.
- K. R. Davidson, S. J. Szarek. Local operator theory, random matrices and
  Banach spaces. *Handbook of the Geometry of Banach Spaces*, 2001.
- I. Halperin. The product of projection operators. *Acta Sci. Math.* 23,
  1962.
- R. A. Horn, C. R. Johnson. *Topics in Matrix Analysis*, 1991, Thm 3.3.16.
- P.-Å. Wedin. Perturbation bounds in connection with singular value
  decomposition. *BIT* 12, 1972.
- M. Miani, H. Roy, S. Hauberg. Bayes without underfitting. AISTATS 2025.
