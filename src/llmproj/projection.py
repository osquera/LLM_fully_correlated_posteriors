"""Alternating projections onto the kernel of the loss-Jacobian (paper Sec. 4).

The posterior is q(theta) = N(theta_map, alpha^-1 U_L U_L^T) where U_L spans
ker(J^L). Because U_L U_L^T is an orthogonal projection, sampling needs only
matrix-vector products with it -- the matrix is never instantiated (paper Sec. 3,
"Computational benefits").

Lemma 4.1 gives the projection onto an intersection of per-batch kernels as

    I - P(M^T M) = lim_t  ( prod_b (I - P(M_b^T M_b)) )^t

so we sweep the data repeatedly, applying one cheap S x S projection per batch.

Differences from the reference JAX implementation, all deliberate:
  * eigenvalue cutoff is *relative* (rtol * lambda_max), not a hardcoded 1e-3.
    Loss-gradient scales for an LLM differ from a toy MLP by many orders of
    magnitude, so an absolute threshold silently keeps noise or discards signal.
  * jax_debug_nans is not replicated; it is ruinous at this scale.
  * `acceleration` defaults to True. Convergence rate is prod cos^2(theta_b)
    (Lemma 4.1) and is slow in practice.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.func import jvp, vjp
from tqdm.auto import tqdm

from llmproj.paramspace import ParamSpace


@dataclass
class BatchFactor:
    """Eigendecomposition of one batch's (J_b J_b^T)^+, shape (R_b, R_b)."""

    eigvecs: torch.Tensor
    inv_eigvals: torch.Tensor
    rank: int


def _jjt(f_batch: Callable, theta: dict[str, torch.Tensor], space: ParamSpace, R: int) -> torch.Tensor:
    """Form J J^T (R x R) matrix-free.

    Column j is J (J^T e_j): one VJP followed by one JVP. Costs 2R passes per
    batch, versus P passes to instantiate J -- which is the whole point, since
    R = batch rows (tens) and P = parameters (millions).
    """
    basis = torch.eye(R, device=space.device, dtype=torch.float32)
    cols = []
    for j in range(R):
        _, vjp_fn = vjp(f_batch, theta)
        jt_e = vjp_fn(basis[j].to(space.dtype))[0]
        _, jjt_e = jvp(f_batch, (theta,), (jt_e,))
        cols.append(jjt_e.float())
    return torch.stack(cols, dim=1)


def precompute_batch_pinv(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    rtol: float = 1e-6,
    progress: bool = True,
) -> list[BatchFactor]:
    """Precompute (J_b J_b^T)^+ for every batch, once, before any sampling.

    These factors are reused by every posterior sample, so this is the
    "preprocessing" column of the paper's Table 1.
    """
    factors: list[BatchFactor] = []
    it = tqdm(batches, desc="precompute (JJ^T)^+", disable=not progress)
    for batch in it:
        f_batch = lambda p: f(p, batch)
        R = int(f_batch(theta).numel())
        jjt = _jjt(f_batch, theta, space, R)
        jjt = 0.5 * (jjt + jjt.T)  # symmetrise away round-off before eigh
        eigvals, eigvecs = torch.linalg.eigh(jjt)

        lam_max = eigvals.max().clamp(min=0.0)
        keep = eigvals > rtol * lam_max
        safe = torch.where(keep, eigvals, torch.ones_like(eigvals))
        inv_eigvals = torch.where(keep, 1.0 / safe, torch.zeros_like(eigvals))

        factors.append(
            BatchFactor(
                eigvecs=eigvecs.to(space.dtype),
                inv_eigvals=inv_eigvals.to(space.dtype),
                rank=int(keep.sum()),
            )
        )
        if progress:
            it.set_postfix(rows=R, rank=factors[-1].rank)
    return factors


def _batch_proj(
    v: torch.Tensor,
    f_batch: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    factor: BatchFactor,
) -> torch.Tensor:
    """v <- (I - J_b^T (J_b J_b^T)^+ J_b) v, one batch, matrix-free."""
    _, jv = jvp(f_batch, (theta,), (space.unflatten(v),))
    w = factor.eigvecs @ (factor.inv_eigvals * (factor.eigvecs.T @ jv))
    _, vjp_fn = vjp(f_batch, theta)
    jt_w = vjp_fn(w)[0]
    return v - space.flatten(jt_w)


@torch.no_grad()
def kernel_residual(
    v: torch.Tensor,
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
) -> float:
    """||J^L v|| / ||v||: how far v still is from ker(J^L). Lower is better.

    This is the correctness diagnostic. If it is not small, the posterior
    samples do *not* preserve the training loss and the method's guarantee
    (Lemma 4.3) does not hold.
    """
    num = 0.0
    for batch in batches:
        f_batch = lambda p: f(p, batch)  # noqa: E731
        _, jv = jvp(f_batch, (theta,), (space.unflatten(v),))
        num += float((jv.float() ** 2).sum())
    return (num ** 0.5) / float(v.norm())


def project_vector(
    v: torch.Tensor,
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    factors: Sequence[BatchFactor],
    n_iterations: int = 500,
    acceleration: bool = True,
    tol: float | None = None,
    check_every: int = 25,
    progress: bool = False,
) -> torch.Tensor:
    """Apply U_L U_L^T to ``v`` by sweeping the data up to ``n_iterations`` times.

    Convergence is linear with rate prod cos^2(theta_b) (Lemma 4.1) and is slow
    in practice -- see the table in README. Measured on a 131-parameter toy
    problem against the exact projector: relative error 2.3e-1 at 20 sweeps,
    4.4e-2 at 200, 3.1e-3 at 600, 7.2e-7 at 2000. Do not assume a handful of
    sweeps suffices; verify with ``kernel_residual``.

    tol: if set, stop early once ||J^L v|| / ||v|| falls below this. Checked
        every ``check_every`` sweeps, costing one extra JVP per batch, which is
        cheap next to the sweep itself and well worth it at LLM scale.
    """
    it = tqdm(range(n_iterations), desc="project", disable=not progress, leave=False)
    for k in it:
        v_start = v
        for batch, factor in zip(batches, factors):
            f_batch = lambda p: f(p, batch)
            v = _batch_proj(v, f_batch, theta, space, factor)
        if acceleration:
            # von Neumann acceleration, as in the reference implementation:
            # extrapolate along the sweep direction. Only pays off after a few
            # hundred sweeps; below that it is a wash.
            d = v_start - v
            dd = d @ d
            if dd > 0:
                t = (v_start @ d) / dd
                v = t * v + (1.0 - t) * v_start
        if tol is not None and (k + 1) % check_every == 0:
            res = kernel_residual(v, f, theta, space, batches)
            if progress:
                it.set_postfix(residual=f"{res:.2e}")
            if res < tol:
                break
    return v


def sample_projected_posterior(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    n_samples: int,
    alpha: float,
    n_iterations: int = 500,
    rtol: float = 1e-6,
    acceleration: bool = True,
    tol: float | None = 1e-3,
    seed: int = 0,
    factors: Sequence[BatchFactor] | None = None,
    progress: bool = True,
):
    """Draw ``n_samples`` from q(theta) = N(theta_map, alpha^-1 U_L U_L^T).

    Returns ``(deltas, info)`` where ``deltas`` is (n_samples, P) holding
    theta - theta_map. Add theta_map to recover weights; keeping the deltas
    makes the diagnostics and the alpha rescaling trivial.
    """
    if factors is None:
        factors = precompute_batch_pinv(f, theta, space, batches, rtol=rtol, progress=progress)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    deltas, residuals, trace_terms = [], [], []

    for i in tqdm(range(n_samples), desc="sampling", disable=not progress):
        eps = torch.randn(space.P, generator=gen, dtype=torch.float32).to(
            device=space.device, dtype=space.dtype
        )
        pv = project_vector(
            eps, f, theta, space, batches, factors,
            n_iterations=n_iterations, acceleration=acceleration, tol=tol,
        )
        # <eps, P eps> is an unbiased Hutchinson estimate of Tr(U_L U_L^T),
        # i.e. of the kernel dimension (paper Lemma 3.4).
        trace_terms.append(float(eps.float() @ pv.float()))
        residuals.append(kernel_residual(pv, f, theta, space, batches))
        deltas.append((pv / (alpha ** 0.5)).cpu())

    info = {
        "kernel_dim_est": sum(trace_terms) / len(trace_terms),
        "kernel_residual_mean": sum(residuals) / len(residuals),
        "kernel_residual_max": max(residuals),
        "batch_ranks": [f.rank for f in factors],
        "P": space.P,
        "alpha": alpha,
        "n_iterations": n_iterations,
    }
    return torch.stack(deltas), info
