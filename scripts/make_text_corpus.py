"""A local text corpus {"texts": [...]} streamed from a Hugging Face dataset,
for constraining a posterior on a pretrained model (no fine-tune: theta_map
is the pretrained weights, the "training data" a sample of text like its
pretraining data).

The first --n documents (in stream order, after --skip) with at least
--min-chars characters. Slice it like a split downstream: train[:512] for the
constraints, train[1000:1256] as held-out text for the fit check.

  uv run python scripts/make_text_corpus.py --out data/fineweb/corpus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--field", default="text")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=600, help="enough for --max-length tokens")
    ap.add_argument("--out", default="data/fineweb/corpus.json")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    texts = []
    for i, r in enumerate(ds):
        if i < args.skip or len(r[args.field]) < args.min_chars:
            continue
        texts.append(r[args.field])
        if len(texts) == args.n:
            break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"texts": texts, "source": vars(args)}))
    print(f"wrote {len(texts)} documents to {out}")


if __name__ == "__main__":
    main()
