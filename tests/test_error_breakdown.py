"""Probes with a missing field must not crash the analysis scripts.

PopQA has a few rows with a null subject; a null name travelled through the
corpus into error_breakdown.py, where ``None in doc`` raised
``TypeError: 'in <string>' requires string as left operand, not NoneType``
after the full posterior had already been sampled (job 32, element 1).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from error_breakdown import answer_spans  # noqa: E402


def probe(**kw):
    p = {"name": "Paris", "freq": 1, "attr": "capital", "answer": " France",
         "prompt": "Q: Where is the Eiffel Tower?\nA:"}
    p.update(kw)
    return p


def test_null_name_popqa_prompt_still_locates_the_answer():
    q = probe(name=None, prompt="Q: Where is the Eiffel Tower?\nA:", answer=" Paris")
    doc = "Q: Where is the Eiffel Tower?\nA: Paris"
    assert answer_spans(doc, q) == [(doc.index("Paris"), doc.index("Paris") + 5)]


def test_null_name_bio_fallback_returns_empty_instead_of_raising():
    assert answer_spans("a document about Paris", probe(name=None)) == []


def test_null_name_absent_from_doc_returns_empty():
    q = probe(name=None, prompt="Q: something else\nA:")
    assert answer_spans("Paris was born in France", q) == []


def test_named_bio_probe_unchanged():
    assert answer_spans("Paris was born in France", probe()) == [(18, 24)]


def test_missing_name_key_is_handled():
    q = probe()
    del q["name"]
    # a probe built by an older corpus may have no "name" at all
    with pytest.raises(KeyError):
        answer_spans("doc", q)
