"""QA-ScaNN on synthetic out-of-distribution data, against plain ScaNN.

    python examples/quickstart.py
"""
import time

import numpy as np

from qa_scann import QAScaNN

rng = np.random.default_rng(0)
n, d, m, nq, k = 50_000, 64, 10_000, 500, 10
X = rng.normal(size=(n, d)).astype(np.float32)
shift = rng.normal(scale=0.8, size=d).astype(np.float32)        # queries come from elsewhere
learn = (rng.normal(size=(m, d)) + shift).astype(np.float32)
Q = (rng.normal(size=(nq, d)) + shift).astype(np.float32)

s = Q @ X.T - 0.5 * (X * X).sum(1)                              # exact top-k under squared L2
gt = np.argsort(-s, axis=1)[:, :k]


def recall(I):
    return np.mean([len(np.intersect1d(a, b)) / k for a, b in zip(I, gt)])


for alpha in (0.0, 0.9):                                         # alpha 0 = plain ScaNN
    t0 = time.time()
    index = QAScaNN(distance="squared_l2", p=2, alpha=alpha, demand_k=k).fit(X, learn)
    built = time.time() - t0
    print("alpha=%.1f  built in %.1fs, %.2f copies/point" % (alpha, built, index.stats["copies_per_point"]))
    for leaves in (1, 2, 4, 8):
        I, _ = index.search_batched(Q, k, leaves_to_search=leaves, reorder=200)
        print("   leaves=%d  recall@%d = %.3f" % (leaves, k, recall(I)))
