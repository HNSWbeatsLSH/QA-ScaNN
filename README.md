# QA-ScaNN

QA-ScaNN is [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
with **query-aware placement**. It is aimed at out-of-distribution workloads,
where the queries do not come from the same distribution as the indexed vectors.

ScaNN partitions the vectors with a k-means tree and stores each one in a single
leaf. A query probes only its nearest leaves, so on such workloads a query's true
neighbours often sit in leaves it never probes. QA-ScaNN takes a sample of
queries (the *learning queries*) and their true neighbours, and stores extra
copies of points in the leaves those queries probe. Points the queries need
often get several copies; the others get none.

Everything else is ScaNN's own: the same tree, codebook, codes and search
(leaf probing, asymmetric-hashing scoring, exact re-ranking). One stage of
ScaNN's build is opened up: its k-means tree can be trained on the dataset
points, as ScaNN does, or on the learning queries.

## How it works

1. **Base index.** A ScaNN index is built in ScaNN's own stages, with these
   defaults:
   - a k-means tree with ⌊√n⌋ leaves, trained by ScaNN's k-means on a sample
     of either
     - the dataset points (`centroids="data"`, ScaNN's default), or
     - the learning queries (`centroids="queries"`);
   - each point assigned to its nearest centre (its "home" leaf);
   - AH codes over 2-dimension blocks, anisotropic threshold 0.2, with the
     codebook trained on that partition;
   - exact re-ranking of a shortlist.

   With query centroids, ScaNN's trained tree goes to its own index factory
   as a pre-trained partitioner, and the factory assigns the points and trains
   the codebook exactly as in a normal build. The codes therefore always match
   the centres. Run on ScaNN's own data-trained tree, this staged build
   reproduces ScaNN's normal build byte for byte.
2. **Demand.** Each learning query is routed to its top-`p` leaves, the way
   the index routes queries. A *demand pair* is a pair (learning query, one of
   its true top-k neighbours). The pair is *served* when that neighbour is
   stored in one of the query's `p` leaves; home leaves serve some pairs for
   free.
3. **Placement.** A greedy min-cost partial cover adds (point, leaf) copies.
   A copy in leaf `j` costs the number of learning queries that probe `j`,
   since each of them will now scan it. The greedy runs until a fraction
   `alpha` of all demand pairs is served.
4. **Storage.** Each point keeps its home leaf and gains the leaves the greedy
   chose. How the copies are coded depends on the distance:
   - **Squared-L2 indexes** use non-residual codes, so every copy of a point
     shares its single code, and the copies add only leaf memberships.
   - **Dot-product indexes** use residual codes, so each copy is encoded
     against its own leaf centre, with ScaNN's anisotropic encoder
     (reimplemented here). On the three inner-product indexes we checked, it
     reproduces ScaNN's own codes for all but 3 of 901,913 points, and each of
     those 3 differs in one block. A point's home code is always ScaNN's own.
5. **Search.** Search is ScaNN's own, with two changes in the patched build:
   it accepts a point in any number of leaves, and it merges a point's copies
   in the shortlist, so results are distinct points.

ScaNN's own spilling ([SOAR](https://arxiv.org/abs/2404.00774)) stores each
point in at most two leaves, both chosen from the geometry of the data alone.
QA-ScaNN instead chooses any number of leaves per point, from the query sample.

## Installation

QA-ScaNN needs a patched ScaNN 1.4.2, which is included in this repository
as `third_party/scann`. The patch (`third_party/scann-multi-token.patch`, 10
files, about 270 lines) lets ScaNN load an index whose points sit in more than
two leaves. The build reports its version as `1.4.2+qa1`. On indexes without
extra copies it returns exactly what stock ScaNN returns. See
[third_party/README.md](third_party/README.md) for the provenance of the copy
and what the patch changes.

The patched ScaNN is built from source. This needs Linux, Clang 19, lld,
Bazel 7.x and Python 3.9–3.13, as for ScaNN itself. On Ubuntu 24.04 the
compiler comes from `apt install clang-19 lld-19 llvm-19 rsync`.

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
| `p` | 4 | leaves per learning query that the placement is optimised for |
| `alpha` | 0.9 | fraction of demand pairs to serve; larger stores more copies, and 0 gives plain ScaNN |
| `centroids` | `"data"` | what ScaNN's k-means tree is trained on: `"data"` (dataset points) or `"queries"` (the learning queries; needs at least `num_leaves` of them) |
| `num_leaves` | ⌊√n⌋ | leaves of the k-means tree |
| `dims_per_block` | 2 | ScaNN AH block size |
| `anisotropic_quantization_threshold` | 0.2 | ScaNN AH threshold |
| `training_sample_size` | 250,000 | ScaNN tree training sample |
| `demand_k` | 100 | neighbours per learning query that count as demand (the k of recall@k) |
| `training_threads` | 1 | threads for ScaNN's tree and codebook training; placement and encoding are single-threaded |

### Search parameters

| parameter | meaning |
|---|---|
| `leaves_to_search` | leaves probed per query |
| `reorder` | size of the AH-scored shortlist that is re-ranked exactly |

### Other methods

- `fit_from_scann(scann_dir, learn, learn_nn)` places copies on an existing
  serialized ScaNN index (one leaf per point) without modifying it.

## VIBE

`vibe/qa_scann/` is the adapter for the [VIBE](https://github.com/vector-index-bench/vibe)
benchmark. It runs on VIBE's out-of-distribution datasets only, since those
ship learning queries; see [vibe/README.md](vibe/README.md).

## Tests

```bash
pip install pytest && pytest tests
```

- `tests/test_core.py` checks two things:
  - at every step, the greedy takes a cell with the best gain-to-cost ratio;
  - the encoder matches a numpy reference byte for byte.
- `tests/test_index.py` builds small indexes end to end, and needs the patched
  ScaNN.

## Verification tools and status

`tools/` contains the scripts that check the build against stock ScaNN:
- `compare_scann_builds.py`: patched and stock ScaNN return the same results;
- `check_staged_build.py`: the staged build equals ScaNN's normal build;
- `check_encoder.py`: the encoder reproduces ScaNN's codes;
- `vibe_summary.py`: summarises VIBE results.

[docs/HANDOVER.md](docs/HANDOVER.md) records the design, the verification
results and the open items.

## License

Apache License 2.0. The ScaNN patch modifies Apache-2.0 ScaNN source; see
[NOTICE](NOTICE).
