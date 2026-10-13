"""End-to-end tests; need the patched ScaNN (scripts/build_scann.sh)."""
import numpy as np
import pytest

from qa_scann import QAScaNN

try:
    from qa_scann.scann_io import require_patched_scann
    require_patched_scann()
except ImportError as e:  # pragma: no cover
    pytest.skip("patched ScaNN not installed: %s" % e, allow_module_level=True)


def _data(seed=0, n=6000, d=32, m=1500, nq=100):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    shift = rng.normal(scale=0.7, size=d).astype(np.float32)   # queries from another distribution
    learn = (rng.normal(size=(m, d)) + shift).astype(np.float32)
    Q = (rng.normal(size=(nq, d)) + shift).astype(np.float32)
    return X, learn, Q


def _gt(Q, X, distance, k):
    s = Q @ X.T
    if distance == "squared_l2":
        s = s - 0.5 * (X * X).sum(1)
    return np.argsort(-s, axis=1)[:, :k]


@pytest.mark.parametrize("distance", ["squared_l2", "dot_product"])
def test_fit_search_save_load(distance, tmp_path):
    X, learn, Q = _data()
    k = 10
    idx = QAScaNN(distance=distance, p=2, alpha=0.95, demand_k=k).fit(X, learn)
    st = idx.stats
    assert st["copies_per_point"] > 1.0 and st["coverage"] >= 0.95 - 1e-9
    gt = _gt(Q, X, distance, k)
    I, D = idx.search_batched(Q, k, leaves_to_search=idx.num_leaves, reorder=2000)
    assert all(len(set(r)) == k for r in I)                       # distinct points
    assert np.mean([len(set(a) & set(b)) / k for a, b in zip(I, gt)]) > 0.99
    for q, row in zip(Q[:10], I[:10]):
        ids, _ = idx.search(q, k, leaves_to_search=idx.num_leaves, reorder=2000)
        np.testing.assert_array_equal(ids, row)
    I1, _ = idx.search_batched(Q, k, 3, 200)
    idx.save(str(tmp_path / "ix"))
    idx2 = QAScaNN.load(str(tmp_path / "ix"))
    I2, _ = idx2.search_batched(Q, k, 3, 200)
    np.testing.assert_array_equal(I1, I2)
    assert idx2.stats == st


@pytest.mark.parametrize("distance", ["squared_l2", "dot_product"])
def test_query_centroids(distance, tmp_path):
    from qa_scann import scann_io as io
    X, learn, Q = _data(seed=2)
    k = 10
    on_data = QAScaNN(distance=distance, p=2, alpha=0.9, demand_k=k).fit(X, learn)
    on_queries = QAScaNN(distance=distance, p=2, alpha=0.9, centroids="queries", demand_k=k).fit(X, learn)
    assert on_queries.stats["centroids"] == "queries"
    # centres trained on the queries sit closer to the queries than data-trained ones
    def gap(ix):
        c = io.read_centers(ix._dir)
        return np.mean(np.min(((learn[:, None, :] - c[None]) ** 2).sum(-1), 1))
    assert gap(on_queries) < gap(on_data)
    gt = _gt(Q, X, distance, k)
    I, _ = on_queries.search_batched(Q, k, leaves_to_search=on_queries.num_leaves, reorder=2000)
    assert all(len(set(r)) == k for r in I)
    assert np.mean([len(set(a) & set(b)) / k for a, b in zip(I, gt)]) > 0.99
    on_queries.save(str(tmp_path / "q"))
    again = QAScaNN.load(str(tmp_path / "q"))
    assert again.centroids == "queries"
    np.testing.assert_array_equal(again.search_batched(Q, k, 3, 200)[0], on_queries.search_batched(Q, k, 3, 200)[0])


def test_query_centroids_need_enough_queries():
    X, learn, _ = _data(seed=3)
    with pytest.raises(ValueError):
        QAScaNN(centroids="queries", num_leaves=200, demand_k=10).fit(X, learn[:100])


def test_placement_beats_plain_scann_at_few_leaves():
    X, learn, Q = _data(seed=1)
    k = 10
    gt = _gt(Q, X, "squared_l2", k)
    rec = lambda I: np.mean([len(set(a) & set(b)) / k for a, b in zip(I, gt)])
    plain = QAScaNN(p=2, alpha=0.0, demand_k=k).fit(X, learn)       # alpha 0: ScaNN as is
    qa = QAScaNN(p=2, alpha=0.95, demand_k=k).fit(X, learn)
    assert plain.stats["copies_per_point"] == 1.0
    assert rec(qa.search_batched(Q, k, 2, 500)[0]) > rec(plain.search_batched(Q, k, 2, 500)[0])
