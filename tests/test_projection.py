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
from llmproj.alpha import estimate_kernel_dim, optimal_alpha

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
