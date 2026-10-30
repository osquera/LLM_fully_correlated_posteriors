"""Load theta_map: a LoRA adapter on the base model, or a fully fine-tuned model.

finetune_lora.py saves either a PEFT adapter directory (adapter_config.json)
or, with --full, the whole model. The scripts take the directory as
--adapter either way; theta is the adapter tensors in the first case and
every parameter in the second.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

LORA_INCLUDE = ["lora_A", "lora_B"]


def is_adapter(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def load_finetuned(base_model: str, path: str | Path, device: str | torch.device = "cpu",
                   dtype: torch.dtype = torch.float32):
    """(model in eval mode on device, default theta include: LoRA tensors or None = all)."""
    # eager attention: torch.func.jvp does not compose with the fused kernels
    if is_adapter(path):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(base_model, attn_implementation="eager", dtype=dtype)
        model, include = PeftModel.from_pretrained(base, str(path)), LORA_INCLUDE
    else:
        model = AutoModelForCausalLM.from_pretrained(str(path), attn_implementation="eager", dtype=dtype)
        include = None
    return model.to(device).eval(), include
