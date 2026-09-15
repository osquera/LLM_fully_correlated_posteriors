"""Closed-form prior precision, paper Lemma 3.4.

    alpha* = ||theta_map||^2 / (P - Tr(I - P(GGN)))

Tr(I - P(GGN)) is the kernel dimension. Since I - P is an orthogonal
projection, eps^T (I - P) eps = ||(I - P) eps||^2, so Hutchinson's estimator
reduces to projecting a few standard normal vectors and reading off
<eps, P eps>. Note P - kernel_dim = rank(J^L), so equivalently

    alpha* = ||theta_map||^2 / rank(J^L)

Unlike the linearised-Laplace marginal likelihood (Eq. 8), this needs no
numerical optimisation. It does, however, divide by a small difference of two
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

    The denominator is ``P - kernel_dim``, a small difference between two large
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
    return theta_norm_sq / max(rank, 1.0)
