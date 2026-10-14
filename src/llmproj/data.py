"""Tokenisation shared by fine-tuning, projection and evaluation.

Which tokens carry loss is a modelling choice, and it must be the same
everywhere: the projection constrains exactly the tokens that fine-tuning fit.

  loss_on="all"       every non-padding token of the text, including the
                      prompt template and the instruction
  loss_on="response"  only the tokens after the response marker (Alpaca's
                      "### Response:"), i.e. what the model is asked to produce

The result carries ``loss_mask`` next to ``attention_mask``: the model always
sees the whole prompt, but only tokens with ``loss_mask == 1`` are targets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

RESPONSE_MARKER = "### Response:"

# The format of tatsu-lab/alpaca's "text" field (reproduces all of train[:2000]
# exactly). Other instruction datasets are put in it to compare like with like.
ALPACA = ("Below is an instruction that describes a task. Write a response that "
          "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
          "### Response:\n")
ALPACA_INPUT = ("Below is an instruction that describes a task, paired with an input that "
                "provides further context. Write a response that appropriately completes "
                "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
                "### Response:\n")


def alpaca_prompt(instruction: str, input: str = "") -> str:
    """The Alpaca prompt, up to and including the response marker."""
    if input and input.strip():
        return ALPACA_INPUT.format(instruction=instruction, input=input)
    return ALPACA.format(instruction=instruction)


def alpaca_text(instruction: str, input: str = "", output: str = "") -> str:
    """A full Alpaca-format example, as in the dataset's "text" field."""
    return alpaca_prompt(instruction, input) + output


LossOn = Literal["all", "response"]


def encode_texts(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    loss_on: LossOn = "all",
    marker: str = RESPONSE_MARKER,
) -> dict[str, torch.Tensor]:
    """Pad/truncate ``texts`` to ``max_length``; returns input_ids,
    attention_mask and loss_mask, each (N, max_length).

    With ``loss_on="response"`` a token is a target when it starts at or after
    the first non-whitespace character following ``marker``. A text without
    the marker raises; a text whose response was truncated away keeps an
    all-zero loss_mask (an exact zero row of J^L, which the projection drops).
    """
    texts = list(texts)
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length",
                    return_tensors="pt", return_offsets_mapping=loss_on == "response")
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if loss_on == "all":
        out["loss_mask"] = enc["attention_mask"].clone()
        return out
    if loss_on != "response":
        raise ValueError(f"unknown loss_on {loss_on!r}")

    starts = []
    for i, t in enumerate(texts):
        k = t.find(marker)
        if k < 0:
            raise ValueError(f"text {i} has no {marker!r}; use loss_on='all' for it")
        k += len(marker)
        while k < len(t) and t[k].isspace():
            k += 1
        starts.append(k)
    tok_start = enc["offset_mapping"][..., 0]  # (N, T) character offsets
    mask = (tok_start >= torch.tensor(starts)[:, None]) & enc["attention_mask"].bool()
    out["loss_mask"] = mask.long()
    return out
