"""Validate the projection against the paper's guarantees.

The decisive test is Lemma 4.3: samples from the loss-projected posterior leave
every training point's loss unchanged to first order, so the loss change must
scale as O(||theta - theta_map||^2) and vanish as alpha grows.
"""

import pytest
import torch
import torch.nn as nn

from llmproj.paramspace import ParamSpace
from llmproj.projection import (
    kernel_residual,
    precompute_batch_pinv,
    project_vector,
    sample_projected_posterior,
)
from llmproj.alpha import (
    estimate_kernel_dim,
    loss_radii,
    loss_radius,
    measure_alpha,
    optimal_alpha,
    per_row_losses,
)

N_ITER = 500  # calibrated: reaches ~3e-3 residual on this toy problem


def _toy(n=40, s=8, d_in=4, n_cls=3, dtype=torch.float64, seed=0):
    """Deterministic toy problem. Seeded per call so tests do not share RNG state."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(d_in, 16), nn.Tanh(), nn.Linear(16, n_cls)).to(dtype)
    space = ParamSpace.from_module(model)
    theta = space.theta(model)
    frozen = space.frozen(model)

    x = torch.randn(n, d_in, dtype=dtype)
    y = torch.randint(0, n_cls, (n,))
    batches = [{"x": x[i : i + s], "y": y[i : i + s]} for i in range(0, n, s)]

    from torch.func import functional_call

    def f(th, batch):
        logits = functional_call(model, {**frozen, **th}, (batch["x"],))
        return nn.functional.cross_entropy(logits, batch["y"], reduction="none")

    return model, space, theta, f, batches


def test_paramspace_roundtrip():
    _, space, theta, _, _ = _toy()
    v = space.flatten(theta)
    assert v.numel() == space.P
    back = space.unflatten(v)
    for name in space.names:
        assert torch.equal(back[name], theta[name])


def test_residual_decreases_with_iterations():
    _, space, theta, f, batches = _toy()
    factors = precompute_batch_pinv(f, theta, space, batches, progress=False)
    gen = torch.Generator().manual_seed(1)
    eps = torch.randn(space.P, generator=gen, dtype=space.dtype)

    residuals = []
    for n_iter in [1, 20, 200, N_ITER]:
        v = project_vector(eps.clone(), f, theta, space, batches, factors,
                           n_iterations=n_iter, acceleration=True)
        residuals.append(kernel_residual(v, f, theta, space, batches))

    assert residuals == sorted(residuals, reverse=True), f"not monotone: {residuals}"
    assert residuals[-1] < 1e-2, f"residual too large: {residuals}"


def test_projection_is_idempotent():
    """U U^T is a projection, so applying it again must not move the vector much."""
    _, space, theta, f, batches = _toy()
    factors = precompute_batch_pinv(f, theta, space, batches, progress=False)
    gen = torch.Generator().manual_seed(2)
    eps = torch.randn(space.P, generator=gen, dtype=space.dtype)
    v1 = project_vector(eps, f, theta, space, batches, factors, n_iterations=N_ITER)
    v2 = project_vector(v1.clone(), f, theta, space, batches, factors, n_iterations=N_ITER)
    rel = float((v2 - v1).norm() / v1.norm())
    assert rel < 1e-2, f"not idempotent: relative change {rel:.3e}"


def test_row_mask_skips_zero_rows_without_changing_the_projection():
    """Masked rows (padding, prompt) are exact zeros of J: skipping them must
    give the same factors' action and the same projected vector."""
    _, space, theta, f0, batches = _toy()
    gen = torch.Generator().manual_seed(5)
    for b in batches:
        b["w"] = (torch.rand(b["y"].shape, generator=gen) > 0.4).to(b["x"].dtype)

    def f(th, batch):
        return f0(th, batch) * batch["w"]

    masked = [{**b, "row_mask": b["w"] > 0} for b in batches]
    eps = torch.randn(space.P, generator=torch.Generator().manual_seed(6), dtype=space.dtype)
    v_all = project_vector(eps.clone(), f, theta, space, batches,
                           precompute_batch_pinv(f, theta, space, batches, progress=False),
                           n_iterations=5)
    v_skip = project_vector(eps.clone(), f, theta, space, masked,
                            precompute_batch_pinv(f, theta, space, masked, progress=False),
                            n_iterations=5)
    # J J^T is formed in float32 either way; eigh on the smaller block rounds
    # differently, so agreement is to float32 precision, not exact
    assert float((v_all - v_skip).norm() / v_all.norm()) < 1e-4


@pytest.mark.parametrize("alpha_scale", [1e2, 1e4, 1e6])
def test_lemma_4_3_loss_preservation(alpha_scale):
    """Loss change must fall ~quadratically in ||delta||, i.e. ~linearly in 1/alpha."""
    from torch.func import functional_call

    model, space, theta, f, batches = _toy()
    factors = precompute_batch_pinv(f, theta, space, batches, progress=False)
    gen = torch.Generator().manual_seed(3)
    eps = torch.randn(space.P, generator=gen, dtype=space.dtype)
    v = project_vector(eps, f, theta, space, batches, factors, n_iterations=N_ITER)
    delta = v / (alpha_scale ** 0.5)

    perturbed = {n: theta[n] + d for n, d in space.unflatten(delta).items()}
    dloss = max(
        float((f(perturbed, b) - f(theta, b)).abs().max()) for b in batches
    )
    # O(||delta||^2) with ||v|| ~ O(sqrt(P)); generous constant, the point is
    # the scaling, checked across the parametrised alphas.
    bound = 50.0 * float(delta.norm()) ** 2
    assert dloss < bound, f"dloss {dloss:.3e} exceeds O(||delta||^2) bound {bound:.3e}"


def test_kernel_dim_and_alpha_are_sane():
    """Hutchinson must recover dim ker(J^L) to within its own sampling error.

    For an orthogonal projection P, Var<eps, P eps> = 2 Tr(P), so the estimator
    from n_probes samples has std sqrt(2 * kernel_dim / n_probes). On this toy
    problem (kernel_dim = 91) that is ~3.4 at n_probes=16 -- visible noise. At
    LLM scale the *relative* error is what matters and it is tiny: a kernel
    dimension of 1e6 with 8 probes has std ~500, i.e. 0.05%.
    """
    n_probes = 16
    _, space, theta, f, batches = _toy()
    factors = precompute_batch_pinv(f, theta, space, batches, progress=False)
    exact = space.P - sum(fa.rank for fa in factors)

    kdim = estimate_kernel_dim(
        f, theta, space, batches, factors, n_probes=n_probes, n_iterations=N_ITER
    )
    assert 0 < kdim < space.P, f"kernel_dim {kdim} outside (0, {space.P})"

    std = (2.0 * exact / n_probes) ** 0.5
    assert abs(kdim - exact) < 4 * std, (
        f"kernel_dim estimate {kdim:.2f} is more than 4 sigma ({4 * std:.2f}) "
        f"from the exact value {exact}"
    )

    alpha = optimal_alpha(theta, space, kdim)
    assert alpha > 0 and torch.isfinite(torch.tensor(alpha))


def test_optimal_alpha_maximises_the_evidence():
    """alpha* must maximise the paper's Eq. 40, (rank/2) log a - a ||theta||^2 / 2.

    The paper's Eq. 42 prints the reciprocal; this pins the corrected form.
    """
    _, space, theta, _, _ = _toy()
    theta_norm_sq = float(sum((t ** 2).sum() for t in theta.values()))
    rank = 17.0
    alpha = optimal_alpha(theta, space, kernel_dim=space.P - rank)
    assert alpha == pytest.approx(rank / theta_norm_sq)

    def log_evidence(a):
        return 0.5 * rank * torch.log(a) - 0.5 * a * theta_norm_sq

    grid = alpha * torch.logspace(-2, 2, 401, dtype=torch.float64)
    best = float(grid[torch.argmax(log_evidence(grid))])
    assert best == pytest.approx(alpha, rel=0.03)


def test_measure_alpha_lands_samples_within_tolerance():
    """alpha from the line search must put samples where the loss tolerance holds."""
    _, space, theta, f, batches = _toy()
    factors = precompute_batch_pinv(f, theta, space, batches, progress=False)
    kdim = float(space.P - sum(fa.rank for fa in factors))
    gen = torch.Generator().manual_seed(3)
    dirs = torch.stack([
        project_vector(torch.randn(space.P, generator=gen, dtype=space.dtype),
                       f, theta, space, batches, factors, n_iterations=N_ITER)
        for _ in range(3)
    ])
    base = per_row_losses(f, theta, batches)
    tol = 0.01 * float(base.mean())
    alpha, radii = measure_alpha(f, theta, space, batches, dirs, kdim, tol)
    assert alpha == pytest.approx(kdim / min(radii) ** 2)

    for d in dirs:
        step = d / d.norm() * min(radii)
        th = {n: theta[n] + v for n, v in space.unflatten(step).items()}
        assert float((per_row_losses(f, th, batches) - base).abs().max()) <= tol


def test_loss_radii_matches_one_search_per_tolerance():
    """The shared grid scan must give what separate line searches give."""
    _, space, theta, f, batches = _toy()
    gen = torch.Generator().manual_seed(5)
    d = torch.randn(space.P, generator=gen, dtype=space.dtype)
    base = per_row_losses(f, theta, batches)
    tols = [x * float(base.mean()) for x in (0.005, 0.02, 0.1)]
    shared = loss_radii(f, theta, space, batches, d, tols, base=base)
    single = [loss_radius(f, theta, space, batches, d, t, base=base) for t in tols]
    assert shared == pytest.approx(single)
    assert shared == sorted(shared)


def test_sample_projected_posterior_end_to_end():
    _, space, theta, f, batches = _toy()
    deltas, info = sample_projected_posterior(
        f, theta, space, batches, n_samples=3, alpha=1e4,
        n_iterations=N_ITER, tol=None, progress=False,
    )
    assert deltas.shape == (3, space.P)
    assert info["kernel_residual_max"] < 1e-2, info
    assert 0 < info["kernel_dim_est"] < space.P


def test_converges_to_exact_projector():
    """The decisive check: alternating projections must reach the true projector.

    Builds J^L explicitly (only tractable because P is tiny here) and compares.
    """
    from torch.func import jvp

    _, space, theta, f, batches = _toy()
    allb = {"x": torch.cat([b["x"] for b in batches]),
            "y": torch.cat([b["y"] for b in batches])}
    rows = []
    for k in range(space.P):
        e = torch.zeros(space.P, dtype=space.dtype)
        e[k] = 1.0
        _, jv = jvp(lambda p: f(p, allb), (theta,), (space.unflatten(e),))
        rows.append(jv)
    JL = torch.stack(rows, dim=1)
    _, sv, Vh = torch.linalg.svd(JL, full_matrices=False)
    rank = int((sv > 1e-10 * sv.max()).sum())
    Vk = Vh[:rank]
    exact_proj = torch.eye(space.P, dtype=space.dtype) - Vk.T @ Vk

    gen = torch.Generator().manual_seed(4)
    eps = torch.randn(space.P, generator=gen, dtype=space.dtype)
    exact = exact_proj @ eps

    factors = precompute_batch_pinv(f, theta, space, batches, progress=False)
    v = project_vector(eps.clone(), f, theta, space, batches, factors, n_iterations=2000)

    cos = float((v @ exact) / (v.norm() * exact.norm()))
    rel = float((v - exact).norm() / exact.norm())
    assert cos > 1 - 1e-6, f"wrong subspace: cos={cos}"
    assert rel < 1e-4, f"not converged to exact projector: rel err {rel:.3e}"


def _reference_project(v, f, theta, space, batches, factors, n_iterations, tol, check_every=25):
    """The one-vector-at-a-time algorithm as it was before batching, verbatim
    apart from its helpers being inlined: the batched code must reproduce it."""
    from torch.func import jvp, vjp

    def residual(v):
        num = sum(float((jvp(lambda p: f(p, b), (theta,), (space.unflatten(v),))[1] ** 2).sum())
                  for b in batches)
        return num ** 0.5 / float(v.norm())

    for k in range(n_iterations):
        v_start = v
        for batch, factor in zip(batches, factors):
            fb = lambda p: f(p, batch)  # noqa: E731
            _, jv = jvp(fb, (theta,), (space.unflatten(v),))
            w = factor.eigvecs @ (factor.inv_eigvals * (factor.eigvecs.T @ jv))
            _, vjp_fn = vjp(fb, theta)
            v = v - space.flatten(vjp_fn(w)[0])
        d = v_start - v
        dd = d @ d
        if dd > 0:
            t = (v_start @ d) / dd
            v = t * v + (1.0 - t) * v_start
        if tol is not None and (k + 1) % check_every == 0 and residual(v) < tol:
            break
    return v


def test_batched_projection_matches_one_vector_at_a_time():
    from llmproj.projection import _jjt, kernel_residuals, project_vectors
    _, space, theta, f, batches = _toy()
    # J J^T: vmapped rows in chunks == one row at a time
    fb = lambda p: f(p, batches[0])  # noqa: E731
    R = int(fb(theta).numel())
    assert torch.allclose(_jjt(fb, theta, space, R, row_batch=3),
                          _jjt(fb, theta, space, R, row_batch=1), rtol=1e-12, atol=1e-12)

    factors = precompute_batch_pinv(f, theta, space, batches, progress=False)
    gen = torch.Generator().manual_seed(0)
    V = torch.randn(3, space.P, generator=gen, dtype=torch.float64)
    # a loose tol so the rows stop at different sweeps: the per-row early stop
    # must then match too
    for tol in (None, 3e-2):
        ref = torch.stack([_reference_project(v.clone(), f, theta, space, batches, factors,
                                              n_iterations=100, tol=tol) for v in V])
        out = project_vectors(V, f, theta, space, batches, factors, n_iterations=100, tol=tol)
        # equal up to round-off (batched matmuls sum in another order, and the
        # acceleration step amplifies it a little): ~1e-10 here, in float64
        assert torch.allclose(out, ref, rtol=1e-7, atol=1e-8), float((out - ref).abs().max())
    res = kernel_residuals(out, f, theta, space, batches)
    assert torch.allclose(res, torch.tensor([kernel_residual(v, f, theta, space, batches)
                                             for v in out], dtype=res.dtype))


def test_sampler_result_does_not_depend_on_vec_batch():
    _, space, theta, f, batches = _toy()
    factors = precompute_batch_pinv(f, theta, space, batches, progress=False)
    kw = dict(n_samples=4, alpha=1.0, n_iterations=60, tol=None, seed=3, factors=factors,
              progress=False)
    d_all, info_all = sample_projected_posterior(f, theta, space, batches, **kw)
    d_one, info_one = sample_projected_posterior(f, theta, space, batches, vec_batch=1, **kw)
    assert torch.allclose(d_all, d_one, rtol=1e-7, atol=1e-8), float((d_all - d_one).abs().max())
    assert abs(info_all["kernel_dim_est"] - info_one["kernel_dim_est"]) < 1e-6


def _dense_jacobian(f, theta, space, batches):
    from torch.func import jacrev
    rows = []
    for b in batches:
        jac = jacrev(lambda th: f(th, b))(theta)
        rows.append(torch.cat([jac[n].reshape(jac[n].shape[0], -1) for n in space.names], 1))
    return torch.cat(rows)


def test_top_singular_subspace_matches_dense_svd():
    from llmproj.lowrank import project_out, top_singular_subspace
    _, space, theta, f, batches = _toy()
    J = _dense_jacobian(f, theta, space, batches)  # (40, 131), rank 40
    s = torch.linalg.svdvals(J)
    out = top_singular_subspace(f, theta, space, batches, k=8, n_iter=6, vec_batch=5,
                                progress=False)
    # leading Ritz values = squared top singular values
    assert torch.allclose(out["sigma2"][:4].double(), s[:4] ** 2, rtol=1e-4)
    # unbiased Frobenius estimate, from only 8 probes: within ~3 std
    assert abs(out["frob2_est"] / float((J ** 2).sum()) - 1) < 1.5
    # k >= rank: the subspace is the whole row space, so projecting it out is
    # the exact kernel projection
    full = top_singular_subspace(f, theta, space, batches, k=40, n_iter=2, progress=False)
    eps = torch.randn(3, space.P, dtype=torch.float64)
    Vh = torch.linalg.svd(J, full_matrices=False).Vh
    exact = eps - (eps @ Vh.T) @ Vh
    assert torch.allclose(project_out(eps, full["U"]), exact, atol=1e-8)
