# Handover

This document records where QA-ScaNN stands: what it is, how it is built, what
has been verified, what is still open, and how to continue. Everything it
refers to is in this repository, except the original research pipeline whose
outputs some checks were compared against (§4).

Status as of 2026-10-14.

## 1. Scope

QA-ScaNN is ScaNN with query-aware placement. Points get extra copies in the
leaves that a sample of queries (the *learning queries*) probes. It is built
on plain ScaNN, not on ScaNN's SOAR variant. The repository has two uses:

1. **A black-box library:** `pip install` it, then call
   `QAScaNN(...).fit(X, learn, learn_nn)` and `.search(...)`.
2. **A method for the [VIBE](https://github.com/vector-index-bench/vibe)
   benchmark**, following VIBE's
   [new method guide](https://github.com/vector-index-bench/vibe/blob/main/docs/NEW_METHOD.md).
   It runs on VIBE's out-of-distribution datasets only, because only those ship
   learning queries.

Decisions made along the way:

- **ScaNN base only.**
- **OOD datasets only.** The adapter has `fit_ood` and no `fit`.
- **Where the centroids come from is an option** (`centroids="data"` or
  `"queries"`). This was done by opening ScaNN's build into its stages (§3).
- **The repository is self-contained.** The patched ScaNN source is vendored
  in `third_party/scann`.

## 2. Repository map

| path | contents |
|---|---|
| `src/qa_scann/index.py` | `QAScaNN`: `fit`, `fit_from_scann`, `search`, `search_batched`, `save`, `load`, `stats` |
| `src/qa_scann/scann_io.py` | readers and writers for a serialized ScaNN index (config, leaf centres, AH codebook, assets) |
| `src/qa_scann/_core.cpp` | pybind11 extension: the placement greedy and ScaNN's AH encoder, single-threaded |
| `third_party/scann/` | ScaNN 1.4.2 source (google-research@634a79f) with the multi-token patch applied |
| `third_party/scann-multi-token.patch`, `third_party/README.md` | the exact diff against upstream, provenance, and how to regenerate |
| `scripts/build_scann.sh` | builds and installs the patched ScaNN wheel from `third_party/scann` |
| `vibe/qa_scann/` | the three VIBE files: `image.def`, `module.py`, `config.yml` |
| `vibe/README.md` | how to submit to VIBE |
| `vibe/pilot/config.yml` | full-grid VIBE config (both centroid sources) used for the comparison in §5 |
| `tests/` | `test_core.py` (no ScaNN needed), `test_index.py` (needs the patched ScaNN) |
| `tools/` | verification scripts (§4) |
| `examples/quickstart.py` | end-to-end example on synthetic data |

## 3. Design

### Build (`QAScaNN.fit`)

1. **Base index.** A ScaNN index of X is built with these defaults:
   - ⌊√n⌋ leaves;
   - AH codes over 2-dimension blocks, anisotropic threshold 0.2;
   - a training sample of 250,000;
   - exact re-ranking.

   ScaNN's distance is `squared_l2` or `dot_product`.
   - `centroids="data"`: `scann_ops_pybind.builder(X).tree().score_ah().reorder().build()`,
     ScaNN's normal build.
   - `centroids="queries"` (`_build_on_query_centroids`):
     1. a tree-only ScaNN build of the learning queries trains the k-means
        tree;
     2. its `serialized_partitioner.pb`, X and the builder's config for X go
        to ScaNN's own factory (`ScannNumpy(dir, assets)` with a `PARTITIONER`
        and a `DATASET_NPY` asset);
     3. the factory assigns each point to its nearest centre and trains the AH
        codebook on that partition.

     This factory path has no thread pool, so it is serial.
2. **Routing.** The base index is serialized. QA-ScaNN reads its centres, home
   leaves and codebook, and routes each learning query to its top-p leaves by
   the index's own routing distance.
3. **Demand and greedy.** Demand pairs are (learning query, one of its top-k
   true neighbours). The greedy (`_core.greedy`) adds (point, leaf) copies.
   Each copy costs the number of learning queries probing that leaf. The
   greedy stops at the shortest prefix that serves a fraction `alpha` of all
   pairs.
4. **Storage.** The index is written in the patched ScaNN's multi-leaf format:
   - `datapoint_to_token_multi.npy` and `datapoint_to_token_offsets.npy`:
     CSR by point, home leaf first, then the copies in greedy order;
   - codes:
     - **squared L2** (non-residual codes): copies share the point's code,
       and the stock `hashed_dataset.npy` is kept;
     - **dot product** (residual codes): each extra copy is encoded against
       its own leaf (`_core.ah_encode_residual`), while home codes stay
       ScaNN's own (`hashed_dataset_multi.npy`).
5. **Search.** Search is ScaNN's own. The patched build merges duplicate
   copies in the shortlist (mean score; over-retrieval ×2), so results are
   distinct points.

### VIBE adapter

`QAScaNN(metric, centroids, num_leaves, p, dims_per_block, alpha)` maps VIBE's
metrics as follows. (`vibe/pilot/config.yml` was written for the v0.1.0
adapter, `QAScaNN(metric, p, alpha, centroids)`; run it with that tag's
`vibe/qa_scann/module.py`.)

| metric | ScaNN distance |
|---|---|
| `normalized`, `cosine` (rows and queries normalised) | squared L2 |
| `ip` | dot product |
| `euclidean` | squared L2 (supported, but VIBE has no OOD Euclidean set) |

The adapter has only `fit_ood`. On an in-distribution dataset VIBE's runner
raises "QAScaNN supports only OOD datasets".

## 4. Verification record

Unless stated otherwise, the checks ran in an environment built from VIBE's
`environment.yml` (Python 3.11, numpy 2.4.4), with the patched ScaNN built by
the build script.

| # | check | result | reproduce with |
|---|---|---|---|
| 1 | clean from-source build of the patched ScaNN: from a fresh upstream checkout + patch, and again from the vendored `third_party/scann`, which is byte-identical to that | both built and installed; 1,862 compile actions, no cache, 59 s and 85 s on 48 cores | `scripts/build_scann.sh` |
| 2 | patched vs PyPI `scann==1.4.2` on stock indexes (llama-128-ip, coco-nomic-768-normalized, yandex-200-cosine; 4 settings each) | identical ids and distances | `tools/compare_scann_builds.py` |
| 3 | unit tests: greedy takes a best ratio cell each step; encoder == numpy reference; fit/search/save/load for both distances and both centroid sources | 18 passed, with ScaNN built from `third_party/scann` | `pytest tests` |
| 4 | greedy port vs the original faiss-based greedy on identical inputs (llama, yi, coco, cqadupstack-lemur, laion) | identical selection order and coverage curves | research pipeline, not in this repo |
| 5 | encoder vs ScaNN's own codes (llama, yi, cqadupstack-lemur; 901,913 points) | identical for all but 3 points, each differing in 1 of 1,024 blocks; the original numpy encoder has the same 3 differences | `tools/check_encoder.py` |
| 6 | full placement vs the indexes measured in the research pipeline, same routes (llama, yi, coco) | tokens, offsets and re-encoded codes byte-identical | research pipeline |
| 7 | same, with QA-ScaNN's own routing (5 datasets) | recall@100 within 3e-5 of the measured sweeps; routing differs only by float32 near-ties on 1–142 queries | research pipeline |
| 8 | staged build fed ScaNN's own data-trained tree vs the normal build (llama-128-ip, coco-nomic-768-normalized, yandex-200-cosine) | all 7 index files byte-identical | `tools/check_staged_build.py` |
| 9 | VIBE harness, `run.py --local`, llama-128-ip, data centroids, 10 builds × 54 settings | completed; builds 11–15 s single-threaded; no duplicate ids; recall within a median 0.002–0.012 of the measured sweeps, with no systematic sign | VIBE + `tools/vibe_summary.py` |
| 10 | VIBE refuses in-distribution datasets | "QAScaNN supports only OOD datasets" | `vibe.runner.build_index(..., None, None)` |
| 11 | data vs query centroids, VIBE harness, `vibe/pilot/config.yml` (both sources × p {1,2,4,8,16} × α {0.9,0.99}, 54 settings, best of 2 runs, 3 datasets run side by side on separate cores) | all 60 builds completed. Best QPS, queries/data: llama-128-ip 0.98× @0.9, 0.69× @0.95; yi-128-ip 0.51×, 0.53×; coco-nomic-768-normalized 1.74×, 1.70×. Query-centroid winners all have p ≤ 4 where query centroids win | VIBE + `tools/vibe_summary.py` |

Never run: building the Apptainer image itself (`./install.sh --algorithm
qa_scann`). Each of its steps was exercised outside a container.

## 5. Open items

1. **Query centroids on the larger OOD sets.** The comparison (§4, check 11)
   covered only the three small OOD sets. Query centroids won on
   text-to-image coco and lost on the attention sets. The VIBE grid therefore
   carries both sources: 10 data builds plus 8 query builds with p ≤ 8, 18 in
   total (VIBE wants fewer than 20). The larger OOD sets (laion,
   imagenet-align, yandex, hotpotqa, cqadupstack) have not been run with
   query centroids.
2. **Before submitting to VIBE.** The repository is published at
   https://github.com/HNSWbeatsLSH/QA-ScaNN. Still to do:
   - Tag `v0.1.0`; `vibe/qa_scann/image.def` checks out that tag.
   - Build the image once with Apptainer.
   - Open the VIBE pull request (see `vibe/README.md`). The user has asked
     not to submit yet.
3. **Left out deliberately; decide before adding:**
   - measured results in the README;
   - a citation;
   - an in-distribution `fit()`;
   - the SOAR-based variant.

## 6. How to continue

```bash
# build and install (Ubuntu 24.04: apt install clang-19 lld-19 llvm-19 rsync)
CC=clang-19 CXX=clang++-19 AR=llvm-ar-19 PYTHON=python3 scripts/build_scann.sh
pip install . && pip install pytest && pytest tests

# run the VIBE adapter without containers
git clone https://github.com/vector-index-bench/vibe.git && cd vibe
cp -r ../QA-ScaNN/vibe/qa_scann vibe/algorithms/qa_scann
mkdir -p data   # put e.g. llama-128-ip.hdf5 from VIBE's Hugging Face dataset here
PATH=$(dirname $(which python3)):$PATH taskset -c 0-7 python3 run.py --local \
    --dataset llama-128-ip --algorithm qa-scann --runs 1
python3 ../QA-ScaNN/tools/vibe_summary.py . llama-128-ip
```

## 7. Gotchas

- **VIBE `--local`** starts its workers with `python3` from `PATH`, so the
  environment with `qa_scann` must come first. It also reserves one core per
  NUMA node, so give it at least 2 cores.
- **ScaNN's pybind module** is reached as
  `scann.scann_ops.py.scann_ops_pybind.scann_pybind`, not
  `scann.scann_ops.cc.scann_pybind`.
- **Asset paths** in `scann_assets.pbtxt` may be relative; ScaNN resolves them
  against the index directory.
- **Query centroids** need at least `num_leaves` learning queries.
- **Single-threaded builds** of large inner-product sets are dominated by
  encoding the extra copies: about 20 minutes on cqadupstack-lemur, which
  has 6.5M extra copies at d = 2048.
- **`pgrep -f` / `pkill -f`** match their own command line. When scripting
  around long runs, use a bracketed pattern (`"[r]un.py"`).
