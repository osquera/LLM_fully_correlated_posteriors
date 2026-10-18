"""Uncertainty about the *meaning* of an answer, split into aleatoric and
epistemic parts with posterior samples.

For one question, every posterior sample s (a model) answers it N times. The
S*N answers are mapped to meaning in two ways:

  clusters    answers that entail each other (both ways) share a cluster
              (Kuhn, Gal & Farquhar, 2023). q_s = sample s's distribution
              over clusters, q_bar = their mean.
                total      H(q_bar)
                aleatoric  mean_s H(q_s)     each model alone is spread over
                                             several meanings
                epistemic  total - aleatoric the models disagree about which
                                             meaning (a "semantic MI")
  embeddings  unit-norm sentence embeddings e; the law of total variance
                total      mean ||e - e_bar||^2
                aleatoric  mean_s mean ||e - e_bar_s||^2   (within a sample)
                epistemic  total - aleatoric              (between samples)

Both epistemic estimates are biased upwards with few answers per sample: S
groups of N draws from *one* distribution still differ by chance (plug-in
entropy of N draws is lower than of S*N; group means scatter). ``corrected``
subtracts the same quantity with answers randomly reassigned to groups, which
keeps the pooled answers and removes exactly the between-sample structure.
The MAP has one model, so its true epistemic part is zero; its corrected value
is the estimator's noise floor.
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Sequence

import numpy as np


def entropy(counts: np.ndarray) -> float:
    """Plug-in entropy (nats) of a count or probability vector."""
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def cluster_decomposition(labels: np.ndarray) -> dict[str, float]:
    """labels: (S, N) cluster id of every answer, row s = posterior sample s."""
    labels = np.asarray(labels)
    k = labels.max() + 1
    q = np.stack([np.bincount(row, minlength=k) for row in labels]).astype(float)
    q /= q.sum(1, keepdims=True)
    total = entropy(q.mean(0))
    alea = float(np.mean([entropy(row) for row in q]))
    return {"total": total, "aleatoric": alea, "epistemic": total - alea}


def embedding_decomposition(emb: np.ndarray) -> dict[str, float]:
    """emb: (S, N, d). Total, within-sample and between-sample spread."""
    emb = np.asarray(emb, dtype=float)
    total = float(((emb - emb.reshape(-1, emb.shape[-1]).mean(0)) ** 2).sum(-1).mean())
    alea = float(((emb - emb.mean(1, keepdims=True)) ** 2).sum(-1).mean())
    return {"total": total, "aleatoric": alea, "epistemic": total - alea}


def shuffle_corrected(x: np.ndarray, decompose: Callable[[np.ndarray], dict],
                      n_shuffles: int = 20, seed: int = 0) -> dict[str, float]:
    """``decompose(x)`` plus ``epistemic_corrected``: its epistemic part minus
    the mean epistemic part over random regroupings of the S*N answers."""
    out = decompose(x)
    rng = np.random.default_rng(seed)
    S, N = x.shape[:2]
    flat = x.reshape(S * N, *x.shape[2:])
    null = [decompose(flat[rng.permutation(S * N)].reshape(x.shape))["epistemic"]
            for _ in range(n_shuffles)]
    out["epistemic_null"] = float(np.mean(null))
    out["epistemic_corrected"] = out["epistemic"] - out["epistemic_null"]
    return out


def greedy_clusters(texts: Sequence[str], same: Callable[[list[tuple[int, int]]], list[bool]],
                    candidate: Callable[[int, int], bool] | None = None) -> np.ndarray:
    """Cluster ids for ``texts``: each text joins the first cluster whose
    representative (first member) it is ``same`` as, else opens a new one.

    same       batched test on (i, j) index pairs, e.g. bidirectional entailment
    candidate  cheap pre-filter; pairs it rejects count as different meanings
               (e.g. embedding cosine below a threshold), saving NLI calls
    Identical texts are merged without calling ``same``.
    """
    ids = np.full(len(texts), -1)
    reps: list[int] = []
    first: dict[str, int] = {}
    for i, t in enumerate(texts):
        if t in first:
            ids[i] = ids[first[t]]
            continue
        first[t] = i
        pairs = [(r, i) for r in reps if candidate is None or candidate(r, i)]
        hits = same(pairs) if pairs else []
        match = next((r for (r, _), h in zip(pairs, hits) if h), None)
        if match is None:
            ids[i] = len(reps)
            reps.append(i)
        else:
            ids[i] = ids[match]
    return ids


def normalize_answer(s: str) -> str:
    """SQuAD-style: lower case, no punctuation, articles or extra spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(answer: str, golds: Sequence[str]) -> bool:
    """A gold alias appears (as whole words) in the normalised answer."""
    a = f" {normalize_answer(answer)} "
    return any(g and f" {g} " in a for g in map(normalize_answer, golds))
