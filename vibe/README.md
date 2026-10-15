# QA-ScaNN in VIBE

`qa_scann/` holds the three files VIBE's
[new method guide](https://github.com/vector-index-bench/vibe/blob/main/docs/NEW_METHOD.md)
asks for, ready to be copied to `vibe/algorithms/qa_scann/` of a VIBE checkout.

| file | contents |
|---|---|
| `image.def` | builds on `base.sif`: installs Clang 19, clones this repository, builds the vendored patched ScaNN (`third_party/scann`) with `scripts/build_scann.sh`, then `pip install`s this package |
| `module.py` | `QAScaNN(BaseANN)`, with `__init__(metric, centroids, num_leaves, p, dims_per_block, alpha, avq_threshold, max_mean_copies=None)` (`num_leaves` None: floor(sqrt(n)); `max_mean_copies` needs qa_scann 0.1.1), `fit_ood`, `set_query_arguments(leaves_to_search, reorder)`, `query`, `batch_query`, `get_additional` and `__str__` |
| `config.yml` | `ood: true`, distance `any`. Two algorithms that differ only in the centroid source: `qa-scann-data` and `qa-scann-queries`. `num_leaves` is `null`, so QA-ScaNN uses floor(sqrt(n)) leaves on every dataset. One run group per `p` {1,2,4}; within a group, `dims_per_block` {1,2,4} x `avq_threshold` {NaN,0.1} (ScaNN's anisotropic threshold; VIBE's ScaNN uses `dims_per_block` {2,4} and also tries 0.25, 0.4 and 0.55), at `alpha` 0.95 and `max_mean_copies` 25 (the placement stops at 25 copies per point on average): 18 builds per algorithm. Each build searches `leaves_to_search` = powers of 2 from p to 128 x VIBE's ScaNN `reorder` {100,200,300,400,600,800,1000}: 42 to 56 settings per build, 882 per algorithm |

**Out-of-distribution datasets only.** The placement needs learning queries,
which VIBE passes (`fit_ood(X_train, X_learn, X_learn_neighbors)`) only on
datasets that ship `learn` and `learn_neighbors`. The module therefore has no
`fit()`: on an in-distribution dataset VIBE's runner raises
`QAScaNN supports only OOD datasets`. This is the same pattern as VIBE's
RoarGraph module.

**Metrics.** VIBE's OOD datasets are `normalized`, `cosine` and `ip`. They
use ScaNN's index as follows:

| metric | ScaNN index | codes |
|---|---|---|
| `normalized` | dot product | residual; each copy is encoded against its own leaf centre |
| `cosine` | dot product, after normalising the vectors in `fit_ood` and each query in `query` | residual; each copy is encoded against its own leaf centre |
| `ip` | dot product | residual; each copy is encoded against its own leaf centre |
| `euclidean` | squared L2 (VIBE has no OOD Euclidean set) | non-residual; a point's copies share its code |

On unit vectors the dot product ranks like squared L2, but with a tree ScaNN gives
the dot-product index residual codes (each point stored relative to its leaf
centre), which are much more accurate. On imagenet-align (num_leaves 1024,
p 2) this took QA-ScaNN from 92-189 to 6,980 QPS at recall@100 0.9.

**Centroids.** `centroids` selects what ScaNN's k-means tree is trained on:
`data` (the dataset points, ScaNN's default) or `queries` (`X_learn`). With
`queries`, ScaNN's own index factory still assigns the points and trains the
AH codebook on the resulting partition.

**Builds** are single-threaded, as VIBE requires: ScaNN's training runs with
one thread, and the placement and encoder are serial.

## Submitting

1. **Tag this repository.** Push it to GitHub, tag the release (`v0.1.1`), and
   set `QA_SCANN_REPO` and `QA_SCANN_REF` at the top of
   `qa_scann/image.def` to match.
2. **Add the files to VIBE.** In a fork of VIBE, copy the folder:

   ```sh
   cp -r qa_scann <vibe>/vibe/algorithms/qa_scann
   ```

3. **Add the README row.** In VIBE's README, add a row to the *Algorithms*
   table, alphabetically between PyNNDescent and RoarGraph:

   ```
   | [QA-ScaNN](https://github.com/HNSWbeatsLSH/QA-ScaNN) | 0.1.1 |
   ```

4. **Build and run** (Apptainer or Singularity needed):

   ```sh
   ./install.sh --algorithm qa_scann
   python3 run.py --dataset llama-128-ip --module qa_scann
   ```

   `--module qa_scann` runs both algorithms; `--algorithm qa-scann-data` (or
   `qa-scann-queries`) runs one.

5. **Open the pull request.**
