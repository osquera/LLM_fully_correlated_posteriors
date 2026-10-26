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


def test_load_texts_local_json(tmp_path):
    import json

    from llmproj.data import load_texts
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps({"texts": [f"doc {i}" for i in range(10)]}))
    assert load_texts(str(p), "train") == [f"doc {i}" for i in range(10)]
    assert load_texts(str(p), "train[2:5]") == ["doc 2", "doc 3", "doc 4"]
    assert load_texts(str(p), "train[:3]") == ["doc 0", "doc 1", "doc 2"]


def test_synthbio_frequencies_and_name_controls():
    from collections import Counter

    from llmproj.synthbio import make_corpus, make_people, make_probes
    people = make_people(freqs=(0, 1, 4), per_freq=30, seed=0)
    docs = make_corpus(people, seed=0)
    assert len(docs) == 30 * (0 + 1 + 4)
    names = [p.name for p in people]
    assert len(set(names)) == len(names)
    # each person appears in exactly f documents (full name, every sentence)
    for p in people:
        assert sum(p.name + " " in d or p.name + "'" in d for d in docs) == p.freq
    # never-seen people have first and last names that seen people also use
    seen = [p for p in people if p.freq > 0]
    first = Counter(p.name.split()[0] for p in seen)
    last = Counter(p.name.split()[1] for p in seen)
    unseen = [p for p in people if p.freq == 0]
    familiar = sum(first[p.name.split()[0]] > 0 and last[p.name.split()[1]] > 0 for p in unseen)
    assert familiar == len(unseen)
    probes = make_probes(people)
    assert len(probes) == len(people) * 4
    city = next(q for q in probes if q["attr"] == "city")
    assert city["prompt"].endswith("was born in") and city["answer"].startswith(" ")
    # deterministic
    assert make_corpus(make_people((0, 1, 4), 30, 0), 0) == docs
