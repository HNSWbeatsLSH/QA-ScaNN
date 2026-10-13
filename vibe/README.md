# QA-ScaNN in VIBE

`qa_scann/` holds the three files VIBE's
[new method guide](https://github.com/vector-index-bench/vibe/blob/main/docs/NEW_METHOD.md)
asks for, ready to be copied to `vibe/algorithms/qa_scann/` of a VIBE checkout.

| file | contents |
|---|---|
| `image.def` | builds on `base.sif`: installs Clang 19, clones this repository, builds the vendored patched ScaNN (`third_party/scann`) with `scripts/build_scann.sh`, then `pip install`s this package |
| `module.py` | `QAScaNN(BaseANN)`, with `__init__(metric, p, alpha, centroids)`, `fit_ood`, `set_query_arguments(leaves_to_search, reorder)`, `query`, `batch_query`, `get_additional` and `__str__` |
| `config.yml` | `ood: true`, distance `any`; 18 builds and 54 search settings (`leaves_to_search` x `reorder`) each. The builds are two run groups: data centroids with `p` {1,2,4,8,16} x `alpha` {0.9,0.99} (10), and query centroids with `p` {1,2,4,8} x `alpha` (8) |

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
| `normalized` | squared L2 | non-residual; a point's copies share its code |
| `cosine` | squared L2, after normalising the vectors in `fit_ood` and each query in `query` | non-residual; a point's copies share its code |
| `ip` | dot product | residual; each copy is encoded against its own leaf centre |

**Centroids.** `centroids` selects what ScaNN's k-means tree is trained on:
`data` (the dataset points, ScaNN's default) or `queries` (`X_learn`). With
`queries`, ScaNN's own index factory still assigns the points and trains the
AH codebook on the resulting partition.

**Builds** are single-threaded, as VIBE requires: ScaNN's training runs with
one thread, and the placement and encoder are serial.

## Submitting

1. **Tag this repository.** Push it to GitHub, tag the release (`v0.1.0`), and
   set `QA_SCANN_REPO` and `QA_SCANN_REF` at the top of
   `qa_scann/image.def` to match.
2. **Add the files to VIBE.** In a fork of VIBE, copy the folder:

   ```sh
   cp -r qa_scann <vibe>/vibe/algorithms/qa_scann
   ```

3. **Add the README row.** In VIBE's README, add a row to the *Algorithms*
   table, alphabetically between PyNNDescent and RoarGraph:

   ```
   | [QA-ScaNN](https://github.com/HNSWbeatsLSH/QA-ScaNN) | 0.1.0 |
   ```

4. **Build and run** (Apptainer or Singularity needed):

   ```sh
   ./install.sh --algorithm qa_scann
   python3 run.py --dataset llama-128-ip --algorithm qa-scann
   ```

5. **Open the pull request.**
