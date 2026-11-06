"""A knowledge-injection corpus from TOFU in the bio corpus format (E42).

TOFU (Maini et al. 2024, locuslab/TOFU, "full") holds 200 fictitious authors
with 20 question-answer pairs each, in consecutive blocks of 20; the base
model cannot know any of them. As in the biographies (E17/E28), authors get
an exposure f, the number of times each of their trained pairs is seen:

  f = 0          --n-unseen authors, never trained
  f = 1/2/4/8    --counts authors each (--freqs), --n-train of their pairs
                 trained f times, --n-heldout more held out

Probes are "Question: <q>\\nAnswer:" -> " <answer>". A trained pair is probed
with its own question (freq = f). A held-out pair of a seen author has
freq = 0, like an unseen author's, with author_seen = True: the author is
known, the fact is not (TOFU's answers overlap within an author, so these
are partly answerable). Every training occurrence is one document. Answers
are sentences, so evaluate_bio.py's exact match is replaced by the share of
answer tokens the MAP gets right (--correct-frac there).

Probes carry name = the author (the fold unit of the logistic test) and
attr = the pair's index within the author's 20.

  uv run python scripts/make_tofu_corpus.py --seed 1 --out data/tofu1/corpus.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

PER_AUTHOR = 20


def qa_prompt(q: str) -> str:
    return f"Question: {q}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--counts", type=int, nargs="+", default=[60, 40, 20, 10],
                    help="seen authors per frequency")
    ap.add_argument("--n-unseen", type=int, default=40, help="authors never trained")
    ap.add_argument("--n-train", type=int, default=5, help="trained pairs per seen author")
    ap.add_argument("--n-heldout", type=int, default=5,
                    help="probed, untrained pairs per author (unseen authors: all probes)")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/tofu/corpus.json")
    args = ap.parse_args()
    if len(args.counts) != len(args.freqs):
        raise SystemExit("--counts needs one count per --freqs")

    ds = load_dataset("locuslab/TOFU", "full", split="train")
    n_auth = len(ds) // PER_AUTHOR
    if args.n_unseen + sum(args.counts) > n_auth:
        raise SystemExit(f"{args.n_unseen + sum(args.counts)} authors asked, {n_auth} in TOFU")
    rng = random.Random(args.seed)
    authors = list(range(n_auth))
    rng.shuffle(authors)

    def pairs(a):
        idx = list(range(a * PER_AUTHOR, (a + 1) * PER_AUTHOR))
        rng.shuffle(idx)
        return idx

    def probe(i, a, f, seen):
        r = ds[i]
        return {"name": f"author_{a}", "freq": f, "attr": i - a * PER_AUTHOR,
                "author_seen": seen, "prompt": qa_prompt(r["question"]),
                "answer": " " + r["answer"].strip()}

    probes, texts = [], []
    for a in (authors.pop() for _ in range(args.n_unseen)):
        probes += [probe(i, a, 0, False) for i in pairs(a)[: args.n_train + args.n_heldout]]
    for f, n in zip(args.freqs, args.counts):
        for a in (authors.pop() for _ in range(n)):
            idx = pairs(a)
            for i in idx[: args.n_train]:
                probes.append(probe(i, a, f, True))
                texts += [f"{qa_prompt(ds[i]['question'])} {ds[i]['answer'].strip()}"] * f
            probes += [probe(i, a, 0, True) for i in idx[args.n_train: args.n_train + args.n_heldout]]
    rng.shuffle(texts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"texts": texts, "probes": probes, "config": vars(args)}))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n_tok = [len(tok(t).input_ids) for t in texts]
    held = sum(p["freq"] == 0 and p["author_seen"] for p in probes)
    print(f"{len(probes)} probes {dict(Counter(p['freq'] for p in probes))} "
          f"({held} of the f = 0 ones from seen authors), {len(texts)} documents, "
          f"tokens per document mean {sum(n_tok) / len(n_tok):.0f} max {max(n_tok)}, "
          f"total {sum(n_tok)} -> {out}")


if __name__ == "__main__":
    main()
