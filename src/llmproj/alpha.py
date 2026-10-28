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
    return loss_radii(f, theta, space, batches, direction, [tol], base=base,
                      grid=grid, n_bisect=n_bisect)[0]


@torch.no_grad()
def loss_radii(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    direction: torch.Tensor,
    tols: Sequence[float],
    base: torch.Tensor | None = None,
    grid: torch.Tensor | None = None,
    n_bisect: int = 8,
) -> list[float]:
    """`loss_radius` for several tolerances, sharing one scan of ``grid``.

    The grid is walked once, stopping at the first step that exceeds the
    largest tolerance; each tolerance then bisects its own bracket. That costs
    ``len(grid) + len(tols) * n_bisect`` loss evaluations at most, rather than
    ``len(tols)`` times the grid.
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

    scanned = []  # (step, worst change), in grid order
    for s in grid.tolist():
        scanned.append((s, worst(s)))
        if scanned[-1][1] > max(tols):
            break

    radii = []
    for tol in tols:
        lo, hi = 0.0, None
        for s, w in scanned:
            if w > tol:
                hi = s
                break
            lo = s
        if hi is None:
            radii.append(float(grid[-1]))
            continue
        if lo == 0.0:
            radii.append(0.0)
            continue
        for _ in range(n_bisect):
            mid = (lo * hi) ** 0.5
            if worst(mid) > tol:
                hi = mid
            else:
                lo = mid
        radii.append(lo)
    return radii


def measure_alpha(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    directions: torch.Tensor,
    kernel_dim: float,
    tol: float,
    all_radii: bool = False,
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

    Only the smallest radius matters, so by default the first direction gets
    the full grid scan + bisection and every other one first a single check
    at the current minimum r*: if it stays within ``tol`` there, its radius is
    >= r* and it is done (reported as r*, a lower bound); only otherwise does it
    bisect, below r*. Same alpha as ``all_radii=True`` (which searches every
    direction fully) up to the bisection resolution, at ~1/4 of the loss passes.
    """
    base = per_row_losses(f, theta, batches)
    if all_radii:
        radii = [loss_radius(f, theta, space, batches, d, tol, base=base, **radius_kwargs)
                 for d in directions]
    else:
        radii = [loss_radius(f, theta, space, batches, directions[0], tol, base=base,
                             **radius_kwargs)]
        for d in directions[1:]:
            r_star = min(radii)
            if r_star <= 0.0 or _worst_change(f, theta, space, batches, d, r_star, base) <= tol:
                radii.append(r_star)
                continue
            # below r*: bisect in log space between the grid point under it and r*
            grid = radius_kwargs.get("grid", torch.logspace(-3, 3, 31))
            below = grid[grid < r_star]
            radii.append(_bisect(f, theta, space, batches, d, tol, base,
                                 float(below[-1]) if len(below) else 0.0, r_star,
                                 radius_kwargs.get("n_bisect", 8)))
    r_star = min(radii)
    if r_star <= 0.0:
        raise ValueError(
            f"even the smallest step on the grid changes a row's loss by more than "
            f"tol={tol:.3g}; pass a finer grid or a looser tol"
        )
    return kernel_dim / r_star**2, radii


@torch.no_grad()
def _worst_change(f, theta, space, batches, direction, s, base) -> float:
    u = direction.to(device=space.device, dtype=space.dtype)
    u = u / u.norm()
    th = {n: theta[n] + d for n, d in space.unflatten(u * s).items()}
    return float((per_row_losses(f, th, batches) - base).abs().max())


@torch.no_grad()
def _bisect(f, theta, space, batches, direction, tol, base, lo, hi, n_bisect) -> float:
    """Largest passing step in (lo, hi], hi known to fail; log-space bisection
    as in `loss_radii` (lo = 0: check the grid's bottom first)."""
    if lo <= 0.0:
        lo = 1e-3
        if _worst_change(f, theta, space, batches, direction, lo, base) > tol:
            return 0.0
    for _ in range(n_bisect):
        mid = (lo * hi) ** 0.5
        if _worst_change(f, theta, space, batches, direction, mid, base) > tol:
            hi = mid
        else:
            lo = mid
    return lo
