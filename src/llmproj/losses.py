"""The rows of the loss-Jacobian J^L (paper Eq. 14) for a causal LM.

A row of J^L is the gradient of one *scalar* loss term. What counts as "one
term" is the central modelling choice for a sequence model:

  mode="sequence"  one row per sequence: the mean NLL over its tokens.
                   J^L is (N, P). Matches Eq. 14 literally, cheapest.
  mode="token"     one row per (sequence, token): each token's NLL.
                   J^L is (N*T, P). Closer to the full-Jacobian kernel of
                   Sec. 4, T times more expensive, and a strictly smaller
                   kernel (more constraints => less posterior spread).

Lemma 4.2 gives ker(J) subset of ker(J^L), so either choice still contains the
prediction-preserving kernel; "token" is simply the tighter relaxation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch.func import functional_call


def make_causal_lm_loss(
    model: torch.nn.Module,
    space,
    frozen: dict[str, torch.Tensor],
    mode: Literal["sequence", "token"] = "sequence",
) -> Callable:
    """Build ``f(theta_dict, batch) -> (R,)`` returning per-row losses.

    ``batch`` is a dict with ``input_ids`` (S, T) and optionally
    ``attention_mask``. Labels are the inputs shifted by one, as usual for
    next-token prediction; padding is excluded via the mask.

    The returned callable is pure in ``theta_dict`` so that torch.func.jvp and
    torch.func.vjp can differentiate through it.
    """

    def f(theta: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        params = {**frozen, **theta}
        input_ids = batch["input_ids"]
        kwargs = {"input_ids": input_ids}
        if "attention_mask" in batch:
            kwargs["attention_mask"] = batch["attention_mask"]

        out = functional_call(model, params, args=(), kwargs=kwargs)
        logits = out.logits if hasattr(out, "logits") else out[0]

        # next-token shift
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        S, Tm1, V = logits.shape

        nll = F.cross_entropy(
            logits.reshape(-1, V).float(),
            targets.reshape(-1),
            reduction="none",
        ).view(S, Tm1)

        if "attention_mask" in batch:
            valid = batch["attention_mask"][:, 1:].to(nll.dtype)
        else:
            valid = torch.ones_like(nll)

        if mode == "sequence":
            # one scalar per sequence: mean NLL over its valid tokens
            denom = valid.sum(dim=1).clamp(min=1.0)
            return (nll * valid).sum(dim=1) / denom
        elif mode == "token":
            # one scalar per token; masked positions contribute an exact zero
            # row, which eigh will discard as a null eigenvalue.
            return (nll * valid).reshape(-1)
        else:
            raise ValueError(f"unknown mode {mode!r}")

    return f
