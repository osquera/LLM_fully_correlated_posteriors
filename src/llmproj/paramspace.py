"""Flat <-> dict conversion for the subset of parameters we treat as theta.

The projection operates on a flat vector in R^P. For an LLM, P is almost never
"all parameters": we restrict theta to a subnetwork (LoRA adapters, the last
block, the LM head). Everything outside theta is frozen and passed through to
functional_call unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class ParamSpace:
    """An ordered, flattenable view over a named subset of a module's parameters."""

    def __init__(self, named_params: dict[str, torch.Tensor]):
        if not named_params:
            raise ValueError("ParamSpace got an empty parameter dict")
        self.names: list[str] = list(named_params.keys())
        self.shapes = [tuple(named_params[n].shape) for n in self.names]
        self.numels = [named_params[n].numel() for n in self.names]
        self.P = int(sum(self.numels))
        ref = named_params[self.names[0]]
        self.device = ref.device
        self.dtype = ref.dtype

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        include: Iterable[str] | None = None,
        trainable_only: bool = True,
    ) -> "ParamSpace":
        """Build a space from a module.

        include: if given, keep only parameters whose name contains any of these
            substrings (e.g. ``["lora_A", "lora_B"]`` or ``["layers.29."]``).
        trainable_only: keep only parameters with requires_grad=True. After
            ``peft.get_peft_model`` this already isolates the adapters.
        """
        sel = {}
        for name, p in module.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if include is not None and not any(tok in name for tok in include):
                continue
            sel[name] = p
        return cls(sel)

    def theta(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """Detached current values of theta, as a dict keyed like ``self.names``."""
        src = dict(module.named_parameters())
        return {n: src[n].detach() for n in self.names}

    def frozen(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """Everything functional_call needs that is *not* in theta (params + buffers)."""
        out = {n: p.detach() for n, p in module.named_parameters() if n not in set(self.names)}
        out.update({n: b.detach() for n, b in module.named_buffers()})
        return out

    def flatten(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([d[n].reshape(-1) for n in self.names])

    def unflatten(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        if v.numel() != self.P:
            raise ValueError(f"expected flat vector of size {self.P}, got {v.numel()}")
        out, i = {}, 0
        for n, sh, ne in zip(self.names, self.shapes, self.numels):
            out[n] = v[i : i + ne].view(sh)
            i += ne
        return out

    def randn(self, generator: torch.Generator | None = None, dtype=None) -> torch.Tensor:
        return torch.randn(
            self.P, device=self.device, dtype=dtype or self.dtype, generator=generator
        )

    def __repr__(self) -> str:
        return f"ParamSpace(P={self.P:,}, tensors={len(self.names)}, dtype={self.dtype})"
