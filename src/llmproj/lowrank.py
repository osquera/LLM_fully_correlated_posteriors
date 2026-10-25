"""The top-k right singular subspace of the loss-Jacobian J^L, matrix-free.

The exact kernel projection needs every row of J^L, swept hundreds of times;
at the scale of pretraining data that is unaffordable, and there J^L may not
even have a kernel. The approximation tested here removes only the k
directions J^L is most sensitive to (its top right singular vectors, i.e. the
top eigenvectors of the GGN-like J^T J) and leaves isotropic noise elsewhere:

    delta = (I - U_k U_k^T) eps / sqrt(alpha)

U_k comes from randomized subspace iteration on J^T J, which needs only
products J^T (J V): one JVP and one VJP per batch for a block of vectors, the
same batched passes as the alternating projection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.func import vjp
from tqdm.auto import tqdm

from llmproj.paramspace import ParamSpace
from llmproj.projection import _jvps, _vjps


@torch.no_grad()
def gram_apply(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    V: torch.Tensor,
    vec_batch: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(J^T J V^T)^T for the rows of V (k, P), and ||J v||^2 per row (k,)."""
    out = torch.zeros_like(V)
    sq = torch.zeros(V.shape[0], device=V.device, dtype=torch.float64)
    for batch in batches:
        f_batch = lambda p: f(p, batch)  # noqa: E731
        _, vjp_fn = vjp(f_batch, theta)
        for i in range(0, V.shape[0], vec_batch):
            JV = _jvps(f_batch, theta, space, V[i:i + vec_batch])  # (c, R)
            sq[i:i + vec_batch] += (JV.double() ** 2).sum(1)
            out[i:i + vec_batch] += _vjps(vjp_fn, space, JV.to(space.dtype))
    return out, sq


@torch.no_grad()
def top_singular_subspace(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    k: int,
    n_iter: int = 3,
    vec_batch: int = 64,
    seed: int = 0,
    progress: bool = True,
) -> dict:
    """Randomized subspace iteration for the top-k right singular vectors of J^L.

    Returns ``U`` (k, P) orthonormal rows, sorted by decreasing singular value,
    ``sigma2`` (k,) the Ritz values (squared singular values), and
    ``frob2_est``, an unbiased estimate of ||J^L||_F^2 = Tr(J^T J) from the
    random start, so ``sigma2[:k'].sum() / frob2_est`` is the share of J^L's
    energy in its top k' directions.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Om = torch.randn(k, space.P, generator=gen, dtype=torch.float32).to(space.device, space.dtype)
    frob2 = None
    for it in tqdm(range(n_iter), desc="subspace iteration", disable=not progress):
        Y, sq = gram_apply(f, theta, space, batches, Om, vec_batch)
        if frob2 is None:  # E ||J w||^2 = Tr(J^T J) for standard normal w
            frob2 = float(sq.mean())
        Q, _ = torch.linalg.qr(Y.T)  # (P, k), in the model's dtype
        Om = Q.T.contiguous()
    # Rayleigh-Ritz: eigen-decompose Q^T J^T J Q in the found subspace
    GQ, _ = gram_apply(f, theta, space, batches, Om, vec_batch)
    T = (Om.double() @ GQ.double().T)
    evals, evecs = torch.linalg.eigh(0.5 * (T + T.T))
    order = evals.argsort(descending=True)
    U = (evecs[:, order].T.to(Om.dtype) @ Om)
    U = torch.linalg.qr(U.T)[0].T.contiguous()  # re-orthonormalise
    return {"U": U, "sigma2": evals[order].clamp_min(0).float().cpu(), "frob2_est": frob2}


def project_out(eps: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """(I - U^T U) applied to the rows of eps (n, P), for orthonormal rows U (k, P)."""
    return eps - (eps @ U.T) @ U
