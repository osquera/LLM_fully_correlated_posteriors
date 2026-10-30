"""A knowledge-injection corpus of real facts (PopQA) in the bio corpus format.

The biographies' design (E17/E28) with real entities and real questions:
PopQA questions are drawn without replacement and each is given a training
frequency f (--freqs / --counts; f = 0 = held out). Every training occurrence
is "Q: <question>\\nA: <answer>", packed --per-doc to a document in a fresh
random order, so the same fact is seen in different contexts. The probe is
the question alone, "Q: <question>\\nA:" -> " <answer>". The base model may
already know some facts (popular subjects), which is the realistic case for
fine-tuning on new knowledge.

Probes carry name = subject (the fold unit of the logistic test), attr =
PopQA relation, pop = subject popularity. Output as make_bio_corpus.py, read
by finetune_lora.py / sample_posterior.py (--dataset ... --split train) and
evaluate_bio.py.

  uv run python scripts/make_popqa_corpus.py --seed 1 --out data/popqa1/corpus.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--counts", type=int, nargs="+", default=[400, 800, 800, 300, 150])
    ap.add_argument("--per-doc", type=int, default=3, help="Q/A pairs per document")
    ap.add_argument("--max-answer-tokens", type=int, default=8)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/popqa/corpus.json")
    args = ap.parse_args()
    if len(args.counts) != len(args.freqs):
        raise SystemExit("--counts needs one count per --freqs")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset("akariasai/PopQA", split="test")
    n_q = Counter(ds["question"])
    # a question asked twice has two different answers (e.g. two films "Heat"): drop both
    rows = [r for r in ds if n_q[r["question"]] == 1
            and 0 < len(tok(" " + r["obj"], add_special_tokens=False).input_ids) <= args.max_answer_tokens]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if len(rows) < sum(args.counts):
        raise SystemExit(f"only {len(rows)} usable questions for {sum(args.counts)}")

    probes, pairs = [], []
    for f, n in zip(args.freqs, args.counts):
        for r in (rows.pop() for _ in range(n)):
            probes.append({"name": r["subj"], "freq": f, "attr": r["prop"], "pop": int(r["s_pop"]),
                           "prompt": f"Q: {r['question']}\nA:", "answer": " " + r["obj"]})
            pairs += [f"Q: {r['question']}\nA: {r['obj']}"] * f
    rng.shuffle(pairs)
    texts = ["\n\n".join(pairs[i:i + args.per_doc]) for i in range(0, len(pairs), args.per_doc)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"texts": texts, "probes": probes, "config": vars(args)}))
    n_tok = [len(tok(t).input_ids) for t in texts]
    print(f"{len(probes)} probes {dict(Counter(p['freq'] for p in probes))}, {len(texts)} documents, "
          f"tokens per document mean {sum(n_tok) / len(n_tok):.0f} max {max(n_tok)}, "
          f"total {sum(n_tok)} -> {out}")


if __name__ == "__main__":
    main()
