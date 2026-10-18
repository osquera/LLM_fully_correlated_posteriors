import numpy as np

from llmproj.semantic import (
    cluster_decomposition,
    embedding_decomposition,
    greedy_clusters,
    is_correct,
    shuffle_corrected,
)


def test_cluster_decomposition_separates_the_two_cases():
    # every model split 50/50 over two meanings: all aleatoric
    alea = cluster_decomposition(np.array([[0, 1] * 5, [0, 1] * 5]))
    assert np.isclose(alea["aleatoric"], np.log(2)) and np.isclose(alea["epistemic"], 0)
    # each model sure, of a different meaning: all epistemic
    epi = cluster_decomposition(np.array([[0] * 10, [1] * 10]))
    assert np.isclose(epi["aleatoric"], 0) and np.isclose(epi["epistemic"], np.log(2))


def test_embedding_decomposition_is_total_variance():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(4, 6, 3))
    d = embedding_decomposition(emb)
    flat = emb.reshape(-1, 3)
    assert np.isclose(d["total"], flat.var(0).sum())
    between = ((emb.mean(1) - flat.mean(0)) ** 2).sum(-1).mean()
    assert np.isclose(d["epistemic"], between)


def test_shuffle_correction_removes_finite_sample_bias():
    # one distribution for every group: raw epistemic > 0 by chance, corrected ~ 0
    # (on average: a single draw scatters by ~0.1 nats)
    rng = np.random.default_rng(1)
    runs = [shuffle_corrected(rng.integers(0, 4, size=(8, 10)), cluster_decomposition,
                              n_shuffles=20, seed=i) for i in range(100)]
    assert np.mean([d["epistemic"] for d in runs]) > 0.1
    assert abs(np.mean([d["epistemic_corrected"] for d in runs])) < 0.02
    # genuinely different groups keep a large corrected value
    labels = np.repeat(np.arange(8)[:, None] % 4, 10, axis=1)
    assert shuffle_corrected(labels, cluster_decomposition)["epistemic_corrected"] > 1.0


def test_greedy_clusters_by_meaning():
    texts = ["Paris", "paris, France", "London", "Paris", "the capital is Paris"]
    city = lambda t: "paris" if "paris" in t.lower() else t.lower()
    same = lambda pairs: [city(texts[i]) == city(texts[j]) for i, j in pairs]
    ids = greedy_clusters(texts, same)
    assert list(ids) == [0, 0, 1, 0, 0]
    # a pre-filter that rejects everything leaves only exact duplicates merged
    ids = greedy_clusters(texts, same, candidate=lambda i, j: False)
    assert list(ids) == [0, 1, 2, 0, 3]


def test_is_correct():
    assert is_correct("It was David Seville.", ["David Seville"])
    assert is_correct("The Beatles", ["Beatles"])
    assert not is_correct("Davidson", ["David"])
