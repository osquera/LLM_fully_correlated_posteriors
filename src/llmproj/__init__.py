"""Fully correlated projected posteriors for small LLMs.

Implements the loss-projected posterior of Miani, Roy & Hauberg (2024),
"Bayes without Underfitting: Fully Correlated Deep Learning Posteriors via
Alternating Projections" (arXiv:2410.16901), Sec. 4.1.

The full-Jacobian variant (Sec. 4) is deliberately not implemented: it needs a
per-batch (S*O) x (S*O) inverse, and for a causal LM O = seq_len * vocab_size,
which is ~25M for a single 512-token SmolLM2 sequence.
"""

from llmproj.alpha import estimate_kernel_dim, optimal_alpha
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace
from llmproj.projection import (
    precompute_batch_pinv,
    project_vector,
    sample_projected_posterior,
)

__all__ = [
    "ParamSpace",
    "estimate_kernel_dim",
    "make_causal_lm_loss",
    "optimal_alpha",
    "precompute_batch_pinv",
    "project_vector",
    "sample_projected_posterior",
]
