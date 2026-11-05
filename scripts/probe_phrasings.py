"""MAP probe accuracy under every phrasing of the biographies (E35).

evaluate_bio.py asks each fact in one phrasing, the probe template
("X studied at"). The biographies use three per attribute, chosen at random,
so a fact seen once is often asked in a phrasing the model never saw it in.
Here every fact is asked in all three, teacher-forced exact match as in
evaluate_bio.py, and each (fact, phrasing) is marked seen when that exact
sentence occurs in the corpus. Reported per training frequency:

  probe       the probe template only (= evaluate_bio.py's accuracy)
  seen        phrasings that occur verbatim in training
  unseen      phrasings that do not (the fact only in other words)
  any / all   the fact answered in at least one / every phrasing

  uv run python scripts/probe_phrasings.py --corpus data/bio_err1/corpus.json \\
      --adapter checkpoints/bio_full1 --out figures/phrasings_bio_full1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from evaluate_bio import encode_probes, per_probe
from evaluate_posterior import evaluate_set
from llmproj.models import load_finetuned
from llmproj.paramspace import ParamSpace
from llmproj.synthbio import TEMPLATES, probe_prefix

N_TEMPLATES = 3


def phrasing_probes(corpus: dict) -> list[dict]:
    """Every probe in every template, with whether that sentence is in training."""
    blob = "\n".join(corpus["texts"])
    out = []
    for i, q in enumerate(corpus["probes"]):
        for j in range(len(TEMPLATES[q["attr"]])):
            prompt = probe_prefix(q["attr"], q["name"], j)
            out.append({**q, "prompt": prompt, "fact": i, "template": j,
                        "seen": (prompt + q["answer"]) in blob})
    return out


def summarize(probes: list[dict], correct: np.ndarray) -> dict:
    freq = np.array([q["freq"] for q in probes])
    tmpl = np.array([q["template"] for q in probes])
    seen = np.array([q["seen"] for q in probes])
    fact = np.array([q["fact"] for q in probes])
    by_fact = np.zeros((fact.max() + 1, N_TEMPLATES), dtype=bool)
    by_fact[fact, tmpl] = correct
    fact_freq = np.zeros(fact.max() + 1, dtype=int)
    fact_freq[fact] = freq

    def acc(m):
        return {"acc": float(correct[m].mean()) if m.any() else None, "n": int(m.sum())}

    report = {}
    for f in sorted(set(freq.tolist())):
        m = freq == f
        report[str(f)] = {"probe": acc(m & (tmpl == 0)), "seen": acc(m & seen),
                          "unseen": acc(m & ~seen),
                          "any": float(by_fact[fact_freq == f].any(1).mean()),
                          "all": float(by_fact[fact_freq == f].all(1).mean())}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    probes = phrasing_probes(corpus)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, include = load_finetuned(args.base_model, args.adapter, args.device)
    space = ParamSpace.from_module(model, include=include, trainable_only=False)
    theta, frozen = space.theta(model), space.frozen(model)
    enc = encode_probes(tokenizer, probes)
    _, s, _ = evaluate_set(model, frozen, theta, space, torch.zeros(1, space.P), enc,
                           args.device, args.batch_size)
    correct = per_probe(s["tok"]["correct"].float(), enc).eq(1).numpy()

    report = summarize(probes, correct)
    print(f"{len(probes)} (fact, phrasing) probes, {args.adapter}")
    print("    f   probe    seen  unseen     any     all")
    for f, r in report.items():
        cells = [r["probe"]["acc"], r["seen"]["acc"], r["unseen"]["acc"], r["any"], r["all"]]
        print(f"  {f:>3} " + " ".join("   -   " if c is None else f"{c:7.3f}" for c in cells))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"report": report, "args": vars(args)}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
