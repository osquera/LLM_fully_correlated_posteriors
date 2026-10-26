"""Write the synthetic biography corpus and its probes (llmproj.synthbio).

  uv run python scripts/make_bio_corpus.py --out data/bio/corpus.json

The file holds {"texts": [...], "probes": [...], "people": [...], "config"};
"texts" is what --dataset data/bio/corpus.json --split train feeds to
finetune_lora.py and sample_posterior.py.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from llmproj.synthbio import make_corpus, make_people, make_probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--per-freq", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/bio/corpus.json")
    args = ap.parse_args()
    people = make_people(tuple(args.freqs), args.per_freq, args.seed)
    texts = make_corpus(people, args.seed)
    probes = make_probes(people)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"texts": texts, "probes": probes,
                               "people": [asdict(p) for p in people], "config": vars(args)}))
    print(f"{len(people)} people {dict(Counter(p.freq for p in people))}, {len(texts)} biographies, "
          f"{len(probes)} probes -> {out}")


if __name__ == "__main__":
    main()
