"""Baselines that share the same loss-Jacobian J^L, alpha and theta.

All three posteriors below are Gaussians over the same subnetwork, differing
only in the covariance:

  diagonal Laplace     Sigma = (diag(J^T J) + alpha I)^-1
  linearised Laplace   Sigma = (J^T J + alpha I)^-1          (paper's `lla`)
  projected (ours)     Sigma = alpha^-1 (I - P(J^T J))

Lemma 3.5 bounds the gap between the last two, so implementing `lla` exactly
also gives a way to *verify* that lemma.

A key structural fact for this setting: J^L has only R rows (R = training rows,
not R = N*O), so J^T J is low rank and `lla` is available in closed form via
the SVD of J -- no Kronecker or last-layer approximation needed. When the dense
J (R x P) fits in memory, the exact kernel projector is available the same way,
which is a much cheaper route to the projected posterior than alternating
projections. Alternating projections earn their keep only once R * P does not
fit, e.g. under mode="token" or large N.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.func import vjp
from tqdm.auto import tqdm

from llmproj.paramspace import ParamSpace


def jacobian_memory_gb(n_rows: int, P: int, dtype=torch.float32) -> float:
    return n_rows * P * torch.finfo(dtype).bits / 8 / 1024**3


def build_loss_jacobian(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    max_gb: float = 8.0,
    progress: bool = True,
) -> torch.Tensor:
    """Materialise J^L (R x P) with one VJP per row.

    Raises if the result would exceed ``max_gb``; use the matrix-free
    alternating projection in that case.
    """
    n_rows = int(sum(f(theta, b).numel() for b in batches))
    need = jacobian_memory_gb(n_rows, space.P, space.dtype)
    if need > max_gb:
        raise MemoryError(
            f"dense J^L would need {need:.1f} GB ({n_rows} rows x {space.P:,} params). "
            f"Raise max_gb, reduce rows (fewer sequences / mode='sequence'), "
            f"shrink P, or use the matrix-free projection instead."
        )

    rows = []
    for batch in tqdm(batches, desc="build J^L", disable=not progress):
        f_batch = lambda p: f(p, batch)  # noqa: E731
        R = int(f_batch(theta).numel())
        _, vjp_fn = vjp(f_batch, theta)
        eye = torch.eye(R, device=space.device, dtype=space.dtype)
        for r in range(R):
            rows.append(space.flatten(vjp_fn(eye[r])[0]))
    return torch.stack(rows)


def _svd(J: torch.Tensor, rtol: float = 1e-6):
    U, s, Vh = torch.linalg.svd(J.float(), full_matrices=False)
    keep = s > rtol * s.max()
    return s[keep], Vh[keep]  # (r,), (r, P) right singular vectors


@torch.no_grad()
def diagonal_laplace_samples(
    J: torch.Tensor, alpha: float, n_samples: int, seed: int = 0
) -> torch.Tensor:
    """Sigma = (diag(J^T J) + alpha I)^-1, sampled elementwise.

    The mean-field approximation the paper argues underfits: it puts variance in
    every direction, including those that change the training loss.
    """
    diag = (J.float() ** 2).sum(dim=0)
    std = (1.0 / (diag + alpha)).sqrt()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(n_samples, J.shape[1], generator=gen)
    return eps * std.cpu()


@torch.no_grad()
def linearized_laplace_samples(
    J: torch.Tensor, alpha: float, n_samples: int, seed: int = 0, rtol: float = 1e-6
) -> torch.Tensor:
    """Sigma = (J^T J + alpha I)^-1, exactly, via the SVD of J.

    In the row space of J the eigenvalue is 1/(s_i^2 + alpha); on its orthogonal
    complement it is 1/alpha. So

        Sigma^{1/2} = alpha^{-1/2} I + sum_i [ (s_i^2+alpha)^{-1/2}
                                               - alpha^{-1/2} ] w_i w_i^T

    with w_i the right singular vectors. This is `lla` with a *fully correlated*
    covariance -- no KFAC, no last-layer restriction -- which is what makes it a
    fair comparison against the projected posterior (cf. Lemma 3.5).
    """
    s, W = _svd(J, rtol)  # W: (r, P)
    coef = (s**2 + alpha).rsqrt() - alpha ** -0.5  # (r,)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(n_samples):
        eps = torch.randn(J.shape[1], generator=gen).to(W.device)
        out.append((alpha ** -0.5 * eps + W.T @ (coef * (W @ eps))).cpu())
    return torch.stack(out)


@torch.no_grad()
def exact_projection_samples(
    J: torch.Tensor, alpha: float, n_samples: int, seed: int = 0, rtol: float = 1e-6
) -> torch.Tensor:
    """Sigma = alpha^-1 (I - P(J^T J)) computed exactly from the SVD of J.

    Same posterior the alternating projection targets, but obtained directly.
    Use it to validate the iterative sampler, and prefer it whenever dense J
    fits: it is orders of magnitude cheaper.
    """
    _, W = _svd(J, rtol)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(n_samples):
        eps = torch.randn(J.shape[1], generator=gen).to(W.device)
        out.append(((eps - W.T @ (W @ eps)) / alpha**0.5).cpu())
    return torch.stack(out)
