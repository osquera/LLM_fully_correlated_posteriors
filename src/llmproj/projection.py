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

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.func import jvp, vjp, vmap
from tqdm.auto import tqdm

from llmproj.paramspace import ParamSpace


@dataclass
class BatchFactor:
    """Eigendecomposition of one batch's (J_b J_b^T)^+, shape (R_b, R_b)."""

    eigvecs: torch.Tensor
    inv_eigvals: torch.Tensor
    rank: int


def _unflatten_batch(space: ParamSpace, V: torch.Tensor) -> dict[str, torch.Tensor]:
    """(K, P) -> dict of (K, *shape) views, the batched ``space.unflatten``."""
    out, i = {}, 0
    for n, sh, ne in zip(space.names, space.shapes, space.numels):
        out[n] = V[:, i : i + ne].reshape(V.shape[0], *sh)
        i += ne
    return out


def _flatten_batch(space: ParamSpace, d: dict[str, torch.Tensor]) -> torch.Tensor:
    """dict of (K, *shape) -> (K, P), the batched ``space.flatten``."""
    return torch.cat([d[n].reshape(d[n].shape[0], -1) for n in space.names], dim=1)


def _jvps(f_batch: Callable, theta: dict[str, torch.Tensor], space: ParamSpace,
          V: torch.Tensor) -> torch.Tensor:
    """J V^T for K tangents at once: (K, P) -> (K, R). One vmapped forward-mode
    pass; identical to K separate JVPs, but each GPU pass does K times the work."""
    return vmap(lambda t: jvp(f_batch, (theta,), (t,))[1])(_unflatten_batch(space, V))


def _vjps(vjp_fn: Callable, space: ParamSpace, W: torch.Tensor) -> torch.Tensor:
    """J^T W^T for K cotangents at once: (K, R) -> (K, P)."""
    return _flatten_batch(space, vmap(vjp_fn)(W)[0])


def _jjt(f_batch: Callable, theta: dict[str, torch.Tensor], space: ParamSpace, R: int,
         rows: torch.Tensor | None = None, row_batch: int = 16) -> torch.Tensor:
    """Form J J^T matrix-free, restricted to ``rows`` (default: all R).

    Column j is J (J^T e_j): one VJP followed by one JVP. Costs 2 passes per
    row, versus P passes to instantiate J -- which is the whole point, since
    R = batch rows (tens) and P = parameters (millions). Rows known to be
    exactly zero (masked tokens) need no passes: pass the others as ``rows``
    to get the (len(rows) x len(rows)) block, the only non-zero one.

    The forward pass behind the VJP is shared by all rows, and ``row_batch``
    rows go through each (vmapped) VJP and JVP together.
    """
    rows = torch.arange(R) if rows is None else rows
    rows_dev = rows.to(space.device)
    _, vjp_fn = vjp(f_batch, theta)
    cols = []
    for i in range(0, len(rows), row_batch):
        idx = rows_dev[i : i + row_batch]
        E = torch.zeros(len(idx), R, device=space.device, dtype=space.dtype)
        E[torch.arange(len(idx), device=space.device), idx] = 1.0
        JtE = _vjps(vjp_fn, space, E)  # (c, P)
        cols.append(_jvps(f_batch, theta, space, JtE).float()[:, rows_dev])  # (c, n_rows)
    return torch.cat(cols, dim=0).T


def precompute_batch_pinv(
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    rtol: float = 1e-6,
    progress: bool = True,
    row_batch: int = 16,
) -> list[BatchFactor]:
    """Precompute (J_b J_b^T)^+ for every batch, once, before any sampling.

    These factors are reused by every posterior sample, so this is the
    "preprocessing" column of the paper's Table 1.

    A batch may carry ``row_mask`` (R,), True where a row of J can be
    non-zero; the other rows are exact zeros (padding, or the prompt under a
    response-only loss) and are skipped. The factor is the same either way:
    its eigenvectors are zero on the skipped rows.
    """
    factors: list[BatchFactor] = []
    it = tqdm(batches, desc="precompute (JJ^T)^+", disable=not progress)
    for batch in it:
        f_batch = lambda p: f(p, batch)
        R = int(f_batch(theta).numel())
        rows = None if "row_mask" not in batch else batch["row_mask"].nonzero().squeeze(1).cpu()
        jjt = _jjt(f_batch, theta, space, R, rows, row_batch=row_batch)
        jjt = 0.5 * (jjt + jjt.T)  # symmetrise away round-off before eigh
        eigvals, eigvecs = torch.linalg.eigh(jjt)
        if rows is not None:  # back to all R rows, zero on the skipped ones
            full = torch.zeros(R, eigvecs.size(1), device=eigvecs.device, dtype=eigvecs.dtype)
            full[rows.to(eigvecs.device)] = eigvecs
            eigvecs = full

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
    V: torch.Tensor,
    f_batch: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    factor: BatchFactor,
) -> torch.Tensor:
    """V <- V (I - J_b^T (J_b J_b^T)^+ J_b), one batch, matrix-free, for the K
    rows of V (K, P) at once."""
    JV = _jvps(f_batch, theta, space, V)  # (K, R)
    W = (factor.eigvecs @ (factor.inv_eigvals[:, None] * (factor.eigvecs.T @ JV.T))).T
    _, vjp_fn = vjp(f_batch, theta)
    return V - _vjps(vjp_fn, space, W)


@torch.no_grad()
def kernel_residuals(
    V: torch.Tensor,
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
) -> torch.Tensor:
    """``kernel_residual`` for every row of V (K, P), as a (K,) tensor."""
    num = torch.zeros(V.shape[0], device=V.device, dtype=torch.float64)
    for batch in batches:
        f_batch = lambda p: f(p, batch)  # noqa: E731
        num += (_jvps(f_batch, theta, space, V).double() ** 2).sum(1)
    return (num.sqrt() / V.double().norm(dim=1)).float()


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
    return float(kernel_residuals(v[None], f, theta, space, batches)[0])


def project_vectors(
    V: torch.Tensor,
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
    callback: Callable[[int, Callable[[], torch.Tensor]], None] | None = None,
) -> torch.Tensor:
    """`project_vector` for the K rows of V (K, P) together.

    Every JVP/VJP carries all still-active rows, so a GPU pass does K times
    the work of the one-vector version, for the same result: each row gets
    exactly the sweeps, acceleration steps and early stop it would get alone
    (a row stops being updated once its own residual is below ``tol``).

    callback(k, current) is called after sweep k (1-based); ``current()``
    returns the projection so far (snapshots of the cost curve).
    """
    V = V.clone()
    active = torch.ones(V.shape[0], dtype=torch.bool, device=V.device)
    it = tqdm(range(n_iterations), desc="project", disable=not progress, leave=False)
    for k in it:
        idx = active.nonzero().squeeze(1)
        A = V[idx]
        A_start = A
        for batch, factor in zip(batches, factors):
            f_batch = lambda p: f(p, batch)  # noqa: E731
            A = _batch_proj(A, f_batch, theta, space, factor)
        if acceleration:
            # von Neumann acceleration, as in the reference implementation:
            # extrapolate along the sweep direction, per row. Only pays off
            # after a few hundred sweeps; below that it is a wash.
            d = A_start - A
            dd = (d * d).sum(1)
            t = torch.where(dd > 0, (A_start * d).sum(1) / dd.clamp_min(1e-300),
                            torch.ones_like(dd))
            A = t[:, None] * A + (1.0 - t[:, None]) * A_start
        V[idx] = A
        if callback is not None:
            callback(k + 1, lambda: V.clone())
        if tol is not None and (k + 1) % check_every == 0:
            res = kernel_residuals(A, f, theta, space, batches)
            if progress:
                it.set_postfix(residual=f"{float(res.max()):.2e}", active=int(active.sum()))
            active[idx[res < tol]] = False
            if not active.any():
                break
    return V


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
    return project_vectors(
        v[None], f, theta, space, batches, factors, n_iterations=n_iterations,
        acceleration=acceleration, tol=tol, check_every=check_every, progress=progress,
    )[0]


def project_vectors_pcg(
    V: torch.Tensor,
    f: Callable,
    theta: dict[str, torch.Tensor],
    space: ParamSpace,
    batches: Sequence[dict[str, torch.Tensor]],
    factors: Sequence[BatchFactor],
    n_iterations: int = 100,
    tol: float | None = None,
    progress: bool = False,
    callback: Callable[[int, Callable[[], torch.Tensor]], None] | None = None,
) -> torch.Tensor:
    """(I - P(J^T J)) V^T by preconditioned conjugate gradients, K rows at once.

    The projection is V - J^T x with (J J^T) x = J V: a linear system in the
    (small) row space. CG solves it with one J J^T product per iteration, i.e.
    one VJP and one JVP per batch, the same passes as one alternating sweep.
    The preconditioner is block Jacobi with the per-batch (J_b J_b^T)^+ the
    alternating projections already use: alternating projections are block
    Gauss-Seidel on this system, so this is its Krylov-accelerated relative,
    with sqrt(condition number) convergence instead of linear-rate sweeps.

    The CG residual r = J V - J J^T x = J (V - J^T x) is the kernel residual's
    numerator, so ``tol`` (||J v|| / ||v|| for every row) costs nothing to check.
    """
    K = V.shape[0]
    fbs = [lambda p, b=b: f(p, b) for b in batches]

    def precond(rs):
        return [((r @ fa.eigvecs) * fa.inv_eigvals) @ fa.eigvecs.T for r, fa in zip(rs, factors)]

    def jt(xs):  # J^T x, (K, P)
        out = torch.zeros_like(V)
        for fb, x in zip(fbs, xs):
            _, vjp_fn = vjp(fb, theta)
            out += _vjps(vjp_fn, space, x)
        return out

    def dot(a, b):  # per-row inner product over all batches, (K,) in float64
        return sum((x.double() * y.double()).sum(1) for x, y in zip(a, b))

    with torch.no_grad():
        r = [_jvps(fb, theta, space, V) for fb in fbs]
    x = [torch.zeros_like(ri) for ri in r]
    z = precond(r)
    p = [zi.clone() for zi in z]
    rz = dot(r, z)
    vnorm = V.double().norm(dim=1)
    it = tqdm(range(n_iterations), desc="project (pcg)", disable=not progress, leave=False)
    for k in it:
        u = jt(p)
        with torch.no_grad():
            q = [_jvps(fb, theta, space, u) for fb in fbs]
        pq = dot(p, q)
        a = torch.where(pq > 0, rz / pq.clamp_min(1e-300), torch.zeros_like(pq))
        a_ = a.to(V.dtype)[:, None]
        x = [xi + a_ * pi for xi, pi in zip(x, p)]
        r = [ri - a_ * qi for ri, qi in zip(r, q)]
        z = precond(r)
        rz_new = dot(r, z)
        beta = torch.where(rz > 0, rz_new / rz.clamp_min(1e-300), torch.zeros_like(rz))
        b_ = beta.to(V.dtype)[:, None]
        p = [zi + b_ * pi for zi, pi in zip(z, p)]
        rz = rz_new
        res = dot(r, r).sqrt() / vnorm
        if progress:
            it.set_postfix(residual=f"{float(res.max()):.2e}")
        if callback is not None:
            callback(k + 1, lambda: V - jt(x))
        if tol is not None and bool((res < tol).all()):
            break
    return V - jt(x)


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
    vec_batch: int | None = None,
    row_batch: int = 16,
    solver: str = "alternating",
    snapshots: Sequence[int] = (),
):
    """Draw ``n_samples`` from q(theta) = N(theta_map, alpha^-1 U_L U_L^T).

    Returns ``(deltas, info)`` where ``deltas`` is (n_samples, P) holding
    theta - theta_map. Add theta_map to recover weights; keeping the deltas
    makes the diagnostics and the alpha rescaling trivial.

    vec_batch: samples projected together (default: all); lower it if the
        vmapped passes run out of memory. The draws, and so the result, do not
        depend on it.
    row_batch: rows of J J^T formed together in the precompute.
    solver: "alternating" (sweeps, Lemma 4.1) or "pcg" (`project_vectors_pcg`);
        n_iterations counts sweeps or CG iterations, which cost the same passes.
    snapshots: iteration counts at which to also keep the projection so far:
        info["snapshots"][n] = {"pv": (n_samples, P) at alpha = 1 on the CPU,
        "seconds": projection time up to n, without the snapshotting}.
    """
    if factors is None:
        factors = precompute_batch_pinv(f, theta, space, batches, rtol=rtol,
                                        progress=progress, row_batch=row_batch)

    # the same eps, in the same order, as drawing them one sample at a time
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.stack([torch.randn(space.P, generator=gen, dtype=torch.float32)
                       for _ in range(n_samples)]).to(device=space.device, dtype=space.dtype)
    k = vec_batch or n_samples
    snaps = {n: {"pv": [], "seconds": 0.0} for n in snapshots}
    chunks = []
    for i in tqdm(range(0, n_samples, k), desc="sampling", disable=not progress):
        t0, paused = time.perf_counter(), 0.0

        def keep(n, current):
            nonlocal paused
            if n in snaps:
                t = time.perf_counter()
                with torch.no_grad():
                    snaps[n]["pv"].append(current().cpu())
                snaps[n]["seconds"] += t - t0 - paused
                paused += time.perf_counter() - t

        kw = dict(n_iterations=n_iterations, tol=tol, progress=progress,
                  callback=keep if snaps else None)
        if solver == "pcg":
            chunks.append(project_vectors_pcg(eps[i:i + k], f, theta, space, batches,
                                              factors, **kw))
        else:
            chunks.append(project_vectors(eps[i:i + k], f, theta, space, batches, factors,
                                          acceleration=acceleration, **kw))
    pv = torch.cat(chunks)
    # <eps, P eps> is an unbiased Hutchinson estimate of Tr(U_L U_L^T),
    # i.e. of the kernel dimension (paper Lemma 3.4).
    trace_terms = (eps.float() * pv.float()).sum(1).tolist()
    residuals = kernel_residuals(pv, f, theta, space, batches).tolist()
    deltas = (pv / (alpha ** 0.5)).cpu()

    info = {
        "kernel_dim_est": sum(trace_terms) / len(trace_terms),
        "kernel_residual_mean": sum(residuals) / len(residuals),
        "kernel_residual_max": max(residuals),
        "batch_ranks": [f.rank for f in factors],
        "P": space.P,
        "alpha": alpha,
        "n_iterations": n_iterations,
        "solver": solver,
    }
    if snaps:
        missing = [n for n, sn in snaps.items() if sum(t.shape[0] for t in sn["pv"]) < n_samples]
        if missing:  # an early stop (tol) came first
            print(f"snapshots {missing} not reached (stopped early); dropped")
        info["snapshots"] = {n: {"pv": torch.cat(sn["pv"]), "seconds": sn["seconds"]}
                             for n, sn in snaps.items() if n not in missing}
    return deltas, info
