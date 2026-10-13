"""Unit tests of the native core (no ScaNN needed)."""
import numpy as np
import pytest

from qa_scann import _core


# ------------------------------------------------------------------ greedy

def _instance(seed, n=300, nlist=12, m=200, K=5, p=3):
    rng = np.random.default_rng(seed)
    owner = rng.integers(0, n, size=(m, K)).astype(np.int32)
    route = np.stack([rng.choice(nlist, p, replace=False) for _ in range(m)]).astype(np.int32)
    home = rng.integers(0, nlist, size=n).astype(np.int32)
    return owner, route, home, n, nlist


@pytest.mark.parametrize("seed", range(5))
def test_greedy_takes_a_best_ratio_cell_every_step(seed):
    owner, route, home, n, nlist = _instance(seed)
    m, K = owner.shape
    keys, gains, base = _core.greedy(owner, route, home, n, nlist)
    Qp = np.bincount(route.ravel(), minlength=nlist)
    # demand pairs the home copies leave uncovered, and the cells covering each
    pairs = [(q, owner[q, r]) for q in range(m) for r in range(K)
             if home[owner[q, r]] not in route[q]]
    assert base == m * K - len(pairs)
    cells = {}
    for i, (q, pt) in enumerate(pairs):
        for lst in route[q]:
            cells.setdefault(int(pt) * nlist + int(lst), set()).add(i)
    covered = set()
    for key, g in zip(keys, gains):
        marg = {c: len(s - covered) for c, s in cells.items()}
        best = max(v / Qp[c % nlist] for c, v in marg.items())
        assert marg[int(key)] == g > 0
        assert np.isclose(g / Qp[key % nlist], best)
        covered |= cells[int(key)]
    assert len(covered) == len(pairs) == gains.sum()
    assert len(set(keys.tolist())) == len(keys)
    assert not np.any(home[keys // nlist] == keys % nlist)


def test_greedy_nothing_to_cover():
    owner = np.array([[0, 1]], np.int32)
    route = np.array([[0]], np.int32)
    home = np.zeros(2, np.int32)
    keys, gains, base = _core.greedy(owner, route, home, 2, 3)
    assert len(keys) == 0 and len(gains) == 0 and base == 2


def test_greedy_rejects_bad_ids():
    owner = np.array([[5]], np.int32)
    with pytest.raises(ValueError):
        _core.greedy(owner, np.array([[0]], np.int32), np.zeros(2, np.int32), 2, 3)


# ----------------------------------------------------------------- encoder

def _encode_reference(original, residual, B, t):
    """numpy reference of ScaNN's noise-shaped AH encoder (one chunk)."""
    K, J, w = B.shape
    n, d = residual.shape
    x = original.astype(np.float64)
    r = residual.astype(np.float64)
    idx = np.arange(n)
    nx2 = (x * x).sum(1)
    eta = (t * t / nx2) / ((1 - t * t / nx2) / (d - 1))
    u = (x / np.sqrt(nx2)[:, None]).reshape(n, K, w)
    cand = r.reshape(n, K, w)[:, :, None, :] - B[None]
    en = (cand ** 2).sum(-1)
    ep = (cand * u[:, :, None, :]).sum(-1)
    code = en.argmin(-1)
    order = np.argsort(-en.min(-1), axis=1, kind="stable")
    s = ep[idx[:, None], np.arange(K)[None], code].sum(1)
    for _ in range(10):
        changed = np.zeros(n, bool)
        for i in range(K):
            k = order[:, i]
            cur = code[idx, k]
            new_s = s[:, None] - ep[idx, k, cur][:, None] + ep[idx, k, :]
            par_d = new_s ** 2 - (s ** 2)[:, None]
            cost = eta[:, None] * par_d + ((en[idx, k, :] - en[idx, k, cur][:, None]) - par_d)
            cost[par_d > 0] = np.inf
            cost[idx, cur] = np.inf
            j = cost.argmin(1)
            ok = cost[idx, j] < 0
            code[idx[ok], k[ok]] = j[ok]
            s[ok] = new_s[idx[ok], j[ok]]
            changed |= ok
        if not changed.any():
            break
    return code.astype(np.uint8)


@pytest.mark.parametrize("d,J,w,t", [(16, 16, 2, 0.2), (300, 16, 2, 0.2), (64, 16, 4, 0.3),
                                     (2048, 16, 2, 0.2), (32, 16, 2, float("nan"))])
def test_encoder_matches_numpy_reference(d, J, w, t):
    rng = np.random.default_rng(d + J + w)
    n, nlist = 400, 7
    X = rng.normal(size=(n, d)).astype(np.float32)
    C = rng.normal(scale=0.5, size=(nlist, d)).astype(np.float32)
    rows = rng.integers(0, n, 600).astype(np.int64)
    lists = rng.integers(0, nlist, 600).astype(np.int64)
    B = rng.normal(scale=0.3, size=(d // w, J, w))
    got = _core.ah_encode_residual(X, C, rows, lists, B, t)
    want = _encode_reference(X[rows], X[rows] - C[lists], B, t)
    assert got.shape == (600, d // w)
    np.testing.assert_array_equal(got, want)
