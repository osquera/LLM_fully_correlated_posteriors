"""Closed-form prior precision, paper Lemma 3.4.

    alpha* = ||theta_map||^2 / (P - Tr(I - P(GGN)))

Tr(I - P(GGN)) is the kernel dimension. Since I - P is an orthogonal
projection, eps^T (I - P) eps = ||(I - P) eps||^2, so Hutchinson's estimator
reduces to projecting a few standard normal vectors and reading off
<eps, P eps>. Note P - kernel_dim = rank(J^L), so equivalently

    alpha* = ||theta_map||^2 / rank(J^L)

Unlike the linearised-Laplace marginal likelihood (Eq. 8), this needs no
numerical optimisation.
"""

from __future__ import annotations

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


def optimal_alpha(theta: dict[str, torch.Tensor], space: ParamSpace, kernel_dim: float) -> float:
    """alpha* from Lemma 3.4, given an estimate of the kernel dimension."""
    theta_norm_sq = float(sum((t.float() ** 2).sum() for t in theta.values()))
    rank = max(space.P - kernel_dim, 1.0)
    return theta_norm_sq / rank
