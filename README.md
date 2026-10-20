# QA-ScaNN

QA-ScaNN is [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
with **query-aware point placement**.

ScaNN partitions vectors with a $k$-means tree and stores each one in a single leaf. Queries probe their nearest leaves and return their nearest neighbors within these leaves. QA-ScaNN takes a sample of queries (the *learning queries*) and their true nearest neighbors, and it stores extra copies of points in multiple leaves. The point-leaf assignments are chosen with a greedy approach that seeks to maximize coverage of the learning queries.

QA-ScaNN inherits all other details of ScaNN, including its $k$-means tree, anisotropic quantization method, and exact reranking. We provide an option to train the initial $k$-means tree on the dataset points (as ScaNN does) or on the learning queries; we have generally found the former to be more performant.

ScaNN also offers its own point replication method ([SOAR](https://arxiv.org/abs/2404.00774)) which stores each point in at most two leaves. These assignments are chosen from the geometry of the base vectors alone. QA-ScaNN instead chooses any number of leaves per point and chooses its assignments from query samples.

## Installation

QA-ScaNN needs a patched ScaNN 1.4.2, which is included in this repository
as `third_party/scann`. The patch (`third_party/scann-multi-token.patch`, 10
files, about 270 lines) lets ScaNN load an index whose points sit in more than
two leaves. The build reports its version as `1.4.2+qa1`. On indexes without
extra copies it returns exactly what stock ScaNN returns. See
[third_party/README.md](third_party/README.md) for the provenance of the copy
and what the patch changes.

The patched ScaNN is built from source. This needs Linux, Clang 19, lld,
Bazel 7.x and Python 3.9–3.13, as for ScaNN itself.

```bash
git clone https://github.com/HNSWbeatsLSH/QA-ScaNN.git && cd QA-ScaNN
CC=clang-19 CXX=clang++-19 AR=llvm-ar-19 PYTHON=python3 scripts/build_scann.sh
pip install .
```

`scripts/build_scann.sh` does the following:
- builds the wheel from `third_party/scann` in a scratch directory, leaving the
  repository untouched;
- downloads Bazel 7.6.1 if no `bazel` is on the `PATH`;
- installs TensorFlow 2.20, which ScaNN's build configuration requires, only
  into a throwaway build venv;
- installs the wheel into `$PYTHON`.

Bazel fetches ScaNN's C++ dependencies (abseil, protobuf, pybind11, ...)
during the build.

## Usage

```python
import numpy as np
from qa_scann import QAScaNN

# X: (n, d) float32 corpus; learn: (m, d) sample of queries;
# learn_nn: (m, >= 100) their exact neighbours in X (computed by brute force if omitted)
index = QAScaNN(distance="dot_product", p=4, alpha=0.9, centroids="data")   # or "queries"
index.fit(X, learn, learn_nn)
print(index.stats)        # copies per point, coverage, leaves, ...

ids, dists = index.search(q, k=100, leaves_to_search=10, reorder=1000)
I, D = index.search_batched(Q, k=100, leaves_to_search=10, reorder=1000)

index.save("my_index")
index = QAScaNN.load("my_index")
```

`examples/quickstart.py` runs end to end on synthetic data.

### Parameters

| parameter | default | meaning |
|---|---|---|
| `distance` | `"squared_l2"` | `"squared_l2"` or `"dot_product"`. For cosine similarity, normalise the vectors; on unit vectors both distances rank the same. |
| `p` | 4 | number of probed leaves query that the placement is optimized for |
| `alpha` | 0.9 | target recall; larger `alpha` stores more copies, and 0 gives plain ScaNN |
| `centroids` | `"data"` | what ScaNN's k-means tree is trained on: `"data"` (dataset points) or `"queries"` (the learning queries; needs at least `num_leaves` of them) |
| `num_leaves` | $\lfloor\sqrt{n}\rfloor$ | leaves of the k-means tree |
| `dims_per_block` | 2 | ScaNN asymmetric hashing block size |
| `anisotropic_quantization_threshold` | 0.2 | ScaNN asymmetric hashing threshold |
| `training_sample_size` | 250,000 | ScaNN tree training sample |
| `demand_k` | 100 | neighbours per learning query that count as ``demand'' (the $k$ of recall@$k$) |
| `training_threads` | 1 | threads for ScaNN's tree and codebook training; placement and encoding are single-threaded |

### Search parameters

| parameter | meaning |
|---|---|
| `leaves_to_search` | leaves probed per query; should be at least `p' |
| `reorder` | number of candidates to rerank exactly |

### Other methods

- `fit_from_scann(scann_dir, learn, learn_nn)` places copies on an existing
  serialized ScaNN index (one leaf per point) without modifying it.

## VIBE

`vibe/qa_scann/` is the adapter for the [VIBE](https://github.com/vector-index-bench/vibe)
benchmark. It runs on VIBE's out-of-distribution datasets only, since those
contain learning queries; see [vibe/README.md](vibe/README.md).

## Tests

```bash
pip install pytest && pytest tests
```

- `tests/test_core.py` checks two things:
  - at every step, the greedy approach takes a cell with the best gain-to-cost ratio;
  - the encoder matches a numpy reference byte for byte.
- `tests/test_index.py` builds small indexes end to end, and needs the patched
  ScaNN.

## License

Apache License 2.0. The ScaNN patch modifies Apache-2.0 ScaNN source; see
[NOTICE](NOTICE).
