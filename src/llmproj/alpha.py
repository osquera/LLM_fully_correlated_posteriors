"""Closed-form prior precision, paper Lemma 3.4 (corrected).

    alpha* = (P - Tr(I - P(GGN))) / ||theta_map||^2

The paper prints the reciprocal in Eq. 9 / Eq. 42, but its own stationarity
condition (Eq. 41), -||theta||^2/2 + (P - Tr(I - P))/(2 alpha) = 0, solves to
the form above. It is also MacKay's evidence update alpha = gamma / ||w||^2,
with gamma = rank(J^L) the number of well-determined parameters.

Tr(I - P(GGN)) is the kernel dimension. Since I - P is an orthogonal
projection, eps^T (I - P) eps = ||(I - P) eps||^2, so Hutchinson's estimator
reduces to projecting a few standard normal vectors and reading off
<eps, P eps>. Note P - kernel_dim = rank(J^L), so equivalently

    alpha* = rank(J^L) / ||theta_map||^2

Unlike the linearised-Laplace marginal likelihood (Eq. 8), this needs no
numerical optimisation. It does, however, depend on a small difference of two
large numbers whenever rank(J^L) << P, which is the usual case for a LoRA
adapter: see `optimal_alpha` and README, "The alpha scaling trap".
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import torch

from llmproj.paramspace import ParamSpace
from llmproj.projection import BatchFactor, project_vector


@torch.no_grad()
def estimate_kernel_dim(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    factors: Sequence[BatchFactor],
    n_probes: int = 8,
    n_iterations: int = 500,
    acceleration: bool = True,
    seed: int = 0,
) -> float:
    """Hutchinson estimate of Tr(U_L U_L^T) = dim ker(J^L)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    acc = 0.0
    for _ in range(n_probes):
        eps = torch.randn(space.P, generator=gen, dtype=torch.float32).to(
            device=space.device, dtype=space.dtype
        )
        pv = project_vector(
            eps, f, theta, space, batches, factors,
            n_iterations=n_iterations, acceleration=acceleration, tol=1e-3,
        )
        acc += float(eps.float() @ pv.float())
    return acc / n_probes


def optimal_alpha(
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    kernel_dim: float,
    n_probes: int | None = None,
) -> float:
    """alpha* from Lemma 3.4, given an estimate of the kernel dimension.

    The numerator is ``P - kernel_dim``, a small difference between two large
    numbers. Hutchinson gives ``kernel_dim`` to excellent *relative* accuracy
    and that is beside the point: with std ``sqrt(2 * kernel_dim / n_probes)``,
    the error on the difference is comparable to the difference itself as soon
    as ``rank(J^L) << P``. Pass ``n_probes`` to get a warning when the estimate
    is statistically indistinguishable from a full-rank kernel; see README,
    "The estimator degenerates before the formula does".
    """
    theta_norm_sq = float(sum((t.float() ** 2).sum() for t in theta.values()))
    rank = space.P - kernel_dim
    if n_probes is not None and kernel_dim > 0:
        sigma = (2.0 * kernel_dim / n_probes) ** 0.5
        if rank < 3.0 * sigma:
            warnings.warn(
                f"rank(J^L) = P - kernel_dim = {rank:.1f} is within 3 sigma "
                f"({sigma:.1f}, from {n_probes} probes) of zero, so alpha* is "
                f"not identified by this estimate. Use the exact rank from a "
                f"dense J^L, raise n_probes, or set alpha by hand.",
                RuntimeWarning,
                stacklevel=2,
            )
    return max(rank, 1.0) / theta_norm_sq


@torch.no_grad()
def per_row_losses(
    f: Callable,
    theta: dict[str, torch.Tensor],
    batches: Sequence[dict[str, torch.Tensor]],
) -> torch.Tensor:
    """All rows of ``f`` over ``batches``, concatenated (the rows of J^L)."""
    return torch.cat([f(theta, b).detach().float() for b in batches])


@torch.no_grad()
def loss_radius(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    direction: torch.Tensor,
    tol: float,
    base: torch.Tensor | None = None,
    grid: torch.Tensor | None = None,
    n_bisect: int = 8,
) -> float:
    """Largest step ``s`` along unit ``direction`` with every row's loss within ``tol``.

    Scans ``grid`` (log-spaced step sizes) for the first ``s`` at which
    max_n |l_n(theta + s u) - l_n(theta)| exceeds ``tol``, then bisects in log
    space between it and the last step that passed. Returns ``grid[-1]`` if
    nothing on the grid exceeds ``tol``, and ``0.0`` if even ``grid[0]`` does.
    """
    if base is None:
        base = per_row_losses(f, theta, batches)
    if grid is None:
        grid = torch.logspace(-3, 3, 31)
    u = direction.to(device=space.device, dtype=space.dtype)
    u = u / u.norm()

    def worst(s: float) -> float:
        th = {n: theta[n] + d for n, d in space.unflatten(u * s).items()}
        return float((per_row_losses(f, th, batches) - base).abs().max())

    lo = 0.0
    for s in grid.tolist():
        if worst(s) > tol:
            hi = s
            break
        lo = s
    else:
        return float(grid[-1])
    if lo == 0.0:
        return 0.0
    for _ in range(n_bisect):
        mid = (lo * hi) ** 0.5
        if worst(mid) > tol:
            hi = mid
        else:
            lo = mid
    return lo


def measure_alpha(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    directions: torch.Tensor,
    kernel_dim: float,
    tol: float,
    **radius_kwargs,
) -> tuple[float, list[float]]:
    """Prior precision set from a measured loss tolerance, not from Lemma 3.4.

    For each row of ``directions`` (kernel directions; any scale), find the
    radius at which the worst per-row loss change reaches ``tol``. Take the
    smallest radius r* and return ``alpha = kernel_dim / r*^2``, so that a
    sample of N(0, alpha^-1 U U^T), whose norm concentrates at
    sqrt(kernel_dim / alpha), lands at r*.

    Unlike `optimal_alpha`, this needs only the *relative* accuracy of
    ``kernel_dim``, which Hutchinson provides well at scale.
    """
    base = per_row_losses(f, theta, batches)
    radii = [
        loss_radius(f, theta, space, batches, d, tol, base=base, **radius_kwargs)
        for d in directions
    ]
    r_star = min(radii)
    if r_star <= 0.0:
        raise ValueError(
            f"even the smallest step on the grid changes a row's loss by more than "
            f"tol={tol:.3g}; pass a finer grid or a looser tol"
        )
    return kernel_dim / r_star**2, radii
