"""A bio corpus whose probes ask every fact in all three phrasings (E51).

The paraphrase test of the membership control (E49): a membership score
(familiarity of the question text) can tell trained probes from untrained
ones because each trained fact is asked in a phrasing the model saw. Here
every fact is asked in every template of its attribute; each probe carries
``template``, ``fact`` (index of the original probe) and ``seen`` (that exact
sentence occurs in training, probe_phrasings.py). A trained fact asked in an
unseen phrasing has unfamiliar wording but known content: a score tracking
facts should still call it known, a score tracking wording should not.
The training texts are unchanged, so evaluate_bio.py runs on the output as
on any bio corpus.

  uv run python scripts/make_phrasing_corpus.py data/bio_err1/corpus.json \\
      data/bio_err1/phrasings.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from probe_phrasings import phrasing_probes


def main():
    src, dst = sys.argv[1], sys.argv[2]
    corpus = json.loads(Path(src).read_text())
    probes = phrasing_probes(corpus)
    Path(dst).write_text(json.dumps({**corpus, "probes": probes,
                                     "config": {**corpus.get("config", {}), "phrasings_of": src}}))
    c = Counter((q["freq"] >= 1, q["seen"]) for q in probes)
    print(f"{len(probes)} probes: trained & seen phrasing {c[(True, True)]}, trained & unseen "
          f"phrasing {c[(True, False)]}, untrained {c[(False, False)]} -> {dst}")


if __name__ == "__main__":
    main()
