"""The loss mask must select exactly the response tokens, and nothing else."""

import pytest
import torch

from llmproj.data import encode_texts

PROMPT = ("Below is an instruction that describes a task. Write a response that "
          "appropriately completes the request.\n\n### Instruction:\nName a colour.\n\n"
          "### Response:\n")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", local_files_only=True)
    except OSError:
        pytest.skip("SmolLM2 tokenizer not in the local Hugging Face cache")
    tok.pad_token = tok.pad_token or tok.eos_token
    return tok


def test_response_mask_covers_the_response_only(tokenizer):
    texts = [PROMPT + "Blue is a primary colour.", PROMPT + "Red."]
    enc = encode_texts(tokenizer, texts, max_length=64, loss_on="response")
    for i, t in enumerate(texts):
        ids = enc["input_ids"][i][enc["loss_mask"][i].bool()]
        assert tokenizer.decode(ids) == t[len(PROMPT):]
    assert torch.all(enc["loss_mask"] <= enc["attention_mask"])


def test_all_mask_is_the_attention_mask(tokenizer):
    enc = encode_texts(tokenizer, [PROMPT + "Blue."], max_length=64, loss_on="all")
    assert torch.equal(enc["loss_mask"], enc["attention_mask"])


def test_truncated_response_gives_an_empty_mask(tokenizer):
    enc = encode_texts(tokenizer, [PROMPT + "Blue."], max_length=8, loss_on="response")
    assert enc["loss_mask"].sum() == 0


def test_missing_marker_raises(tokenizer):
    with pytest.raises(ValueError, match="Response"):
        encode_texts(tokenizer, ["plain text, no template"], max_length=16, loss_on="response")
