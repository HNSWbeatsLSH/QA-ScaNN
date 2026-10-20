# QA-ScaNN on VIBE: benchmark run, results, projections, and how to submit

*Status as of 2026-10-19. Run 3 finished on 2026-10-17 at 17:35 UTC. The VIBE
pull request is not opened yet.*

---

## 1. Summary

* **What was run.** QA-ScaNN 0.1.1 (both centroid sources) on VIBE's harness,
  on all 8 out-of-distribution (OOD) datasets. VIBE's own `scann` and
  `ivf(faiss)` ran on the same machine as reference baselines. 46 hours wall
  time in total.
* **Measured.** On our machine, QA-ScaNN with data centroids is faster than
  VIBE's ScaNN at recall@100 = 0.9 on every dataset: 1.3× to 2.2× where VIBE's
  ScaNN reaches 0.9 at all. On yi, llama, lemur and muvera, VIBE's ScaNN never
  reaches 0.9.
* **Forecast.** Scaled to VIBE's published hardware, QA-ScaNN (data
  centroids) would rank on VIBE's current CPU leaderboard at recall 0.9 as
  follows:

  | dataset | rank at recall 0.9 |
  |---|---|
  | yi, llama, imagenet-align, muvera | **#1** |
  | laion, lemur | #1 to 3 (within the forecast uncertainty of the leader) |
  | yandex | #4 |
  | hotpotqa | #14 |

* **Recommendation.** Submit **data centroids only**, as a single algorithm
  `qa-scann` with 18 builds. Keep `dims_per_block = 1`. On VIBE's current
  Xeon hardware, 6 of the 18 configs on cqadupstack-muvera may hit VIBE's
  24-hour timeout, but they don't affect muvera's ranking. Raise this with the
  maintainers in the PR (section 6).

---

## 2. What was run

### Code

| component | version |
|---|---|
| QA-ScaNN | `v0.1.1` = commit `c80499d` on github.com/HNSWbeatsLSH/QA-ScaNN (adds the mean-copies cap, the build-script fix, and the dot-product adapter) |
| patched ScaNN | 1.4.2+qa1, vendored in `third_party/scann`, built by `scripts/build_scann.sh` inside the `qa_scann` image |
| VIBE | upstream `main` at `7ac76c1` (unchanged since), plus `vibe/algorithms/qa_scann/` from QA-ScaNN |
| images | `base.sif`, `qa_scann.sif` (from tag v0.1.1), `scann.sif` (PyPI scann 1.4.2), `faiss.sif` (faiss-cpu 1.14.3) |

### Machine and procedure

* AMD EPYC 9575F (2 × 64 cores, SMT on), 3 TB RAM. The machine was shared with
  other users: about 60–70% of all cores were busy with another user's job when
  the runs began. Each build had its own pinned core, but cores were time-shared.
* VIBE's `run.py`, one dataset at a time, `--runs 1` (one search pass per
  setting; VIBE's default is 5). Workers were pinned to cores 47–63 and
  111–127, one build per core, all single-threaded, as VIBE requires.
* A systemd user service with a hard memory cap of 500 GB. The worker count
  per dataset was chosen from measured per-build memory peaks: 32 on the small
  datasets, down to 3 on muvera.
* `tools/vibe_memwatch.py` logged peak RSS and wall time for every build
  process.

### QA-ScaNN configuration (run 3)

| parameter | values |
|---|---|
| distance | dot product for `ip`, `normalized` and `cosine` (residual codes); squared L2 only for `euclidean` |
| `num_leaves` | `null` = ⌊√n⌋ |
| `p` | {1, 2, 4} (one run group each) |
| `dims_per_block` | {1, 2, 4} |
| `avq_threshold` (ScaNN's anisotropic threshold) | {NaN, 0.1} |
| `alpha` | 0.95 |
| `max_mean_copies` | 25 |
| `leaves_to_search` | powers of 2 from p to 128 |
| `reorder` | {100, 200, 300, 400, 600, 800, 1000}, the same as VIBE's ScaNN |
| centroid source | `data` (algorithm `qa-scann-data`) and `queries` (`qa-scann-queries`) |

That is 18 builds and 882 search settings per algorithm per dataset. See
`docs/SCANN_AND_QA_SCANN.md` for what each parameter does.

### Datasets (VIBE's OOD set)

| dataset | n | d | learning queries | QA leaves (√n) |
|---|---|---|---|---|
| yi-128-ip | 187,843 | 128 | 186,843 | 433 |
| llama-128-ip | 256,921 | 128 | 255,921 | 506 |
| yandex-200-cosine | 1,000,000 | 200 | 1,000,000 | 1,000 |
| imagenet-align-640-normalized | 1,281,167 | 640 | 315,648 | 1,131 |
| laion-clip-512-normalized | 1,000,448 | 512 | 999,448 | 1,000 |
| hotpotqa-harrier-640-normalized | 5,233,329 | 640 | 96,845 | 2,287 |
| cqadupstack-lemur-2048-ip | 457,149 | 2,048 | 1,331,947 | 676 |
| cqadupstack-muvera-5120-ip | 457,149 | 5,120 | 1,331,947 | 676 |

All files were checked against VIBE's published SHA-256 sums.

### How the configuration was arrived at (runs 1 and 2, plus tests)

1. **Dot product instead of squared L2 on unit-norm data.** ScaNN uses
   residual codes only for the dot product. The old squared-L2 mapping capped
   QA-ScaNN's recall on normalized data. On imagenet-align, the switch took
   QA-ScaNN from about 190 to about 7,000 QPS at recall 0.9. With
   `alpha = 0`, the dot-product index matches VIBE's ScaNN, so the remaining
   gain comes from the placement.
2. **Anisotropic threshold.** Across VIBE's sweep, NaN and 0.1 together kept
   99–100% of the best QPS. 0.25, 0.4 and 0.55 never added anything.
3. **`dims_per_block = 1`.** It is needed for recall at or above about 0.96 on
   the 128-dimensional inner-product datasets within VIBE's `reorder` ≤ 1000.
4. **√n leaves.** This kept 89–100% of the best QPS at a third of the builds.
   **`leaves_to_search` ≤ 128.** Higher values were never on the Pareto front
   below recall 0.997.
5. **Copies cap of 25.** It binds only on cqadupstack: on 6 of 18 lemur builds
   and 12 of 18 muvera builds.
6. **Centroid source.** Data centroids beat query centroids everywhere on the
   dot-product path, except for a near-tie on lemur.

---

## 3. Measured results (our machine)

QPS at recall@100 0.90 / 0.95 (exact-ID recall, one search pass). "–" = never
reached. Build time, index and peak RAM are for `qa-scann-data`.

| dataset | **QA-ScaNN data** | QA-ScaNN queries | VIBE ScaNN | IVF (Faiss) | build time median / max | index (max) | peak RAM / build | copies / pt (median) | wall time (workers) |
|---|---|---|---|---|---|---|---|---|---|
| yi | **17,936 / 13,455** | 6,318 / – | – / – | 1,786 / – | 17 s / 27 s | 0.2 GB | 1 GB | 4.8 | 2 min (32) |
| llama | **25,080 / 19,537** | – / – | – / – | 1,679 / – | 16 s / 22 s | 0.2 GB | 1.5 GB | 1.9 | 1.5 min (32) |
| yandex | **6,501 / 4,709** | 3,131 / 1,577 | 5,121 / 2,691 | 985 / 525 | 3.5 / 10.5 min | 2.4 GB | 9 GB | 9.9 | 19 min (30) |
| imagenet-align | **7,609 / 5,018** | 3,638 / 1,805 | 4,655 / 2,728 | 1,027 / 658 | 3.2 / 9.0 min | 4.3 GB | 12 GB | 2.2 | 17 min (21) |
| laion | **3,433 / 2,747** | 171 / – | 1,754 / 737 | 361 / 198 | 11.6 / 33.8 min | 6.6 GB | 20 GB | 12.7 | 1.5 h (13) |
| hotpotqa | **946 / 499** | 533 / 131 | 437 / 222 | 103 / 53 | 10.8 / 21.9 min | 15.6 GB | 40 GB | 1.75 | 1.5 h (7) |
| lemur | **519 / 273** | 387 / 218 | – / – | – / – | 34 / 73 min | 14.5 GB | 52 GB | 21.7 | 8.3 h (4) |
| muvera | **377 / 200** | 126 / 64 | – / – (0.80: 410) | – / – | 96 / 188 min | 36.2 GB | 128 GB | 25.0 (capped) | 33.9 h (3) |

---

## 4. Forecast: QA-ScaNN on VIBE's published leaderboard

### Method

1. **Recall.** VIBE's own definition (`export_results.get_recall_values`): a
   returned point counts if its distance is within the true 100th-neighbour
   distance + 10⁻³. It is averaged over the 1,000 test queries. This is more
   lenient than exact-ID recall, so the starting QPS below is higher than in
   section 3 on several datasets.
2. **Hardware scaling.** VIBE's published numbers come from Intel Xeon Gold
   6230 CPUs. We ran VIBE's own ScaNN (1,470 settings) and IVF (34 settings)
   with identical parameters. The ratio is the median, over matched settings,
   of published QPS ÷ our QPS. It is 0.39–0.61 on the five smaller datasets and
   0.22–0.31 on hotpotqa, lemur and muvera. The forecast is our QA-ScaNN QPS ×
   the ScaNN ratio and × the IVF ratio, which gives the range shown.
3. **Leaderboard.** VIBE's published `summary.parquet` (the data behind
   vector-index-bench.github.io), k = 100, CPU methods only. Each method's best
   QPS among its configurations at or above the target recall.
4. **Check.** Applying the ratios to our ScaNN and IVF reproduces their
   published QPS within 0–20%.

### Forecast (QPS on VIBE's hardware, rank among CPU methods)

| dataset | ratio (ScaNN / IVF) | recall | **QA-ScaNN data** | **rank** | leaders (published QPS) | QA-ScaNN queries |
|---|---|---|---|---|---|---|
| yi | 0.41 / 0.48 | 0.90 | 7,326–8,638 | **#1 / 5** | RoarGraph 4,404 · MLANN-RF 3,761 | #3 |
| | | 0.95 | 5,496–6,480 | **#1 / 4** | MLANN-RF 2,397 | – |
| llama | 0.39 / 0.43 | 0.90 | 9,813–10,805 | **#1 / 5** | MLANN-RF 7,967 · RoarGraph 6,105 | – |
| | | 0.95 | 7,644–8,417 | **#1 / 4** | MLANN-RF 5,906 | – |
| yandex | 0.44 / 0.47 | 0.90 | 3,645–3,905 | #4 / 25 | glass 4,346 · PAG 4,230 · NGT-QG 3,907 | #12–15 |
| | | 0.95 | 2,554–2,737 | #3 / 25 | glass 3,496 · PAG 2,929 | #13–15 |
| imagenet-align | 0.51 / 0.46 | 0.90 | 4,269–4,799 | **#1 / 22** | glass 4,155 · PAG 3,822 | #5–7 |
| | | 0.95 | 2,956–3,324 | #2–3 / 22 | PAG 3,383 · glass 3,074 | #9–11 |
| laion | 0.50 / 0.61 | 0.90 | 2,666–3,230 | #1–3 / 23 | glass 2,765 · PAG 2,756 | #18 |
| | | 0.95 | 1,731–2,098 | #1–2 / 23 | PAG 1,807 · glass 1,573 | #19 |
| hotpotqa | 0.27 / 0.27 | 0.90 | 409–413 | #14 / 23 | SymphonyQG 1,834 · glass 1,828 | #18 |
| | | 0.95 | 238–240 | #14 / 22 | PAG 1,540 · SymphonyQG 1,246 | #18 |
| lemur | 0.25 / 0.22 | 0.90 | 114–129 | #1–3 / 9 | glass 124 · RoarGraph 118 | #3 |
| | | 0.95 | 60–68 | **#1 / 7** | RoarGraph 50 | #1–2 |
| muvera | 0.30 / 0.31 | 0.90 | 113–117 | **#1 / 8** | glass 104 · LoRANN 78 | #4 |
| | | 0.95 | 60–62 | **#1 / 6** | RoarGraph 29 | #2 |

### Caveats

* The forecast assumes QA-ScaNN scales between machines like VIBE's ScaNN,
  which uses the same library. Graph methods may scale differently.
* Our numbers are single search passes on a shared machine (±5–10%); VIBE
  takes the best of 5 on dedicated nodes. Ranks within about 15% of a leader
  (yandex, imagenet-align at 0.95, laion, lemur at 0.90) are uncertain.
* VIBE's next results update moves to AMD Turin 9965 CPUs, the same generation
  as our EPYC, which can reorder close races. No date has been announced; past
  updates were 4–10 months apart (May 2025, Sept 2025, July 2026).
* The forecast covers QPS only. VIBE also reports build time and index size,
  where QA-ScaNN costs more (sections 3 and 5).
* hotpotqa is QA-ScaNN's weak case: 5.2M points against 97k learning queries
  give only about 1.75 copies per point, and graph indexes lead there.

---

## 5. Projected resource use against VIBE's requirements

### VIBE's requirements

| requirement | scope | enforced? |
|---|---|---|
| `run.py --timeout`, default **24 h** | per **run** = one build config: its build plus every search setting × 5 passes × 1,000 queries | **yes**: the process is killed. Settings already finished are kept; the rest are lost. |
| "should take less than 24 hours to run for a given dataset using parallelism > 12" (README) | per dataset, all methods together (about 300 core-h) | no. VIBE's own runs already exceed it on hotpotqa (about 640 core-h). |
| "at least 16 GB of memory per used core" (README); example Slurm job 256 GB / 15 workers | memory of concurrently running configs | no. Not enforced, but a job that runs out loses that config. Published indexes reach 60–76 GB on hotpotqa, lemur and muvera, so VIBE must already give those datasets more memory. |
| fewer than 20 builds and 60 search settings per algorithm; single-threaded builds (`NEW_METHOD.md`) | config | guideline |
| index size | per config | **not limited**, only reported |

**Parallelism does not change the per-config time.** VIBE runs
`--parallelism n` configs at once, each pinned to its own core and
single-threaded. A config therefore takes the same time with 1 or 30 workers.
Parallelism affects only total wall time and memory, because concurrent
configs share the job's memory.

### `qa-scann` (data centroids, 18 builds) projected onto VIBE's current hardware, 5 search passes

| dataset | longest config | 18 builds: build only / build + search | index (max) | peak RAM / build | VIBE's costliest method: total build time |
|---|---|---|---|---|---|
| yi | 0.1 h | 0.2 / 0.8 core-h | 0.2 GB | 1 GB | 1.4 h (FALCONN++) |
| llama | 0.1 h | 0.2 / 0.6 | 0.2 GB | 1.5 GB | 1.8 h (RaBitQ-HNSW) |
| yandex | 0.8 h | 3.0 / 6.0 | 2.4 GB | 9 GB | 11.6 h (RaBitQ-HNSW) |
| imagenet-align | 0.5 h | 2.2 / 4.4 | 4.3 GB | 12 GB | 19.5 h (RaBitQ-HNSW) |
| laion | 2.6 h | 8.7 / 17.8 | 6.6 GB | 20 GB | 17.2 h (RaBitQ-HNSW) |
| hotpotqa | 1.8 h | 9.2 / 16.2 | 15.6 GB | 40 GB | 123 h (RaBitQ-HNSW) |
| lemur | 15.0 h | 36.9 / 119 | 14.5 GB | 52 GB | 53 h (RaBitQ-HNSW) |
| **muvera** | **33.0 h** | 95.5 / 304 | 36.2 GB | 128 GB | 31.3 h (VIBE's ScaNN) |

For comparison, the longest single config of any published method on laion,
hotpotqa, lemur and muvera is 11.5 h. On muvera the costliest published method totals about
34 core-h including search, and indexes there reach up to 76 GB.

* **Timeout.** Everything is under 24 h except muvera, where **6 of 18 configs,
  all with `dims_per_block = 1`, project to 24–33 h**. Searches dominate:
  49 settings × 1,000 queries × 5 passes on hardware that searches about 3.5×
  slower than ours on this dataset.
* **On AMD Turin**, VIBE's next hardware, muvera's longest config would be
  about 10 h and lemur's about 4 h.
* **Memory.** About 128 GB per muvera build and 52 GB per lemur build. Only 1
  or 2 muvera builds fit in a 256 GB job.

### The `dims_per_block = 1` trade-off (forecast rank of data centroids)

| dataset | recall | with dpb {1,2,4} | without dpb 1 |
|---|---|---|---|
| yi | 0.95 / 0.99 | #1 / #1 | #2 / not reached |
| llama | 0.90 / 0.95 / 0.99 | #1 / #1 / #1 | #3 / #2–3 / not reached |
| lemur | 0.90 / 0.95 | #1–3 / #1 | #3 / not reached |
| yandex | 0.99 | #3 | #5 |
| imagenet-align, laion, hotpotqa, muvera | all | unchanged | unchanged |

Dropping `dims_per_block = 1` makes muvera safe on the current hardware
(longest config about 16 h, total 131 core-h), but costs the first places on
yi, llama and lemur. **Muvera's best points use `dims_per_block = 4`**, so if
its 6 slowest configs time out, muvera's ranking is unaffected; VIBE only
wastes about 6 × 24 core-h. The config cannot treat muvera differently from
lemur, yi and llama, because all four are `ip` datasets.

---

## 6. Recommended submission config

One algorithm, `qa-scann`, with data centroids: 18 builds and at most 56
search settings per build, within VIBE's guidelines. Replace
`vibe/qa_scann/config.yml` in QA-ScaNN with this, and copy it into VIBE:

```yaml
float:
  any:
  - base_args: ['@metric', data]
    constructor: QAScaNN
    disabled: false
    ood: true
    singularity_image: qa_scann
    module: vibe.algorithms.qa_scann
    name: qa-scann
    run_groups:
      p1:
        args:
          num_leaves: [null]  # null: QA-ScaNN's default of floor(sqrt(n)) leaves
          p: [1]
          dims_per_block: [1, 2, 4]
          alpha: [0.95]
          avq_threshold: [.nan, 0.1]
          max_mean_copies: [25]  # cap on the mean copies per point
        query_args:
          leaves_to_search: [1, 2, 4, 8, 16, 32, 64, 128]
          reorder: [100, 200, 300, 400, 600, 800, 1000]
      p2:
        args:
          num_leaves: [null]  # null: QA-ScaNN's default of floor(sqrt(n)) leaves
          p: [2]
          dims_per_block: [1, 2, 4]
          alpha: [0.95]
          avq_threshold: [.nan, 0.1]
          max_mean_copies: [25]  # cap on the mean copies per point
        query_args:
          leaves_to_search: [2, 4, 8, 16, 32, 64, 128]
          reorder: [100, 200, 300, 400, 600, 800, 1000]
      p4:
        args:
          num_leaves: [null]  # null: QA-ScaNN's default of floor(sqrt(n)) leaves
          p: [4]
          dims_per_block: [1, 2, 4]
          alpha: [0.95]
          avq_threshold: [.nan, 0.1]
          max_mean_copies: [25]  # cap on the mean copies per point
        query_args:
          leaves_to_search: [4, 8, 16, 32, 64, 128]
          reorder: [100, 200, 300, 400, 600, 800, 1000]
```

To make muvera safe on VIBE's current hardware instead, use
`dims_per_block: [2, 4]` in all three groups (12 builds) and accept the rank
losses in section 5.

`module.py` and `image.def` need no change: the adapter in v0.1.1 already takes
`(metric, centroids, num_leaves, p, dims_per_block, alpha, avq_threshold,
max_mean_copies)`, and `image.def` installs tag v0.1.1. **No new QA-ScaNN
release is needed.** The config and adapter go to VIBE through the PR, and the
image only installs the library.

---

## 7. How to complete the submission

The local VIBE clone is `/mnt/data/ivf_proj/vibe`. Its `origin` is your fork
github.com/apadaki/vibe and `upstream` is vector-index-bench/vibe. The branch
`add-qa-scann` already holds `vibe/algorithms/qa_scann/` (the
two-algorithm config) and the README row, uncommitted. There is no git
identity on the machine, so pass one per command (`-c user.name=… -c
user.email=…`).

### Step 1: update the config in QA-ScaNN and commit

```sh
cd /mnt/data/ivf_proj/QA-ScaNN
# write the YAML from section 6 to vibe/qa_scann/config.yml
# update vibe/README.md: one algorithm `qa-scann` (data centroids), 18 builds, 882 search settings
git add vibe/qa_scann/config.yml vibe/README.md docs/SCANN_AND_QA_SCANN.md docs/VIBE_RUN_AND_SUBMISSION.md
git -c user.name="Ashwin Padaki" -c user.email="apadaki@engineering.upenn.edu" \
    commit -m "VIBE config: data centroids only (qa-scann), 18 builds; run report and ScaNN notes"
git push origin main          # no new tag needed: the image installs the library from v0.1.1
```

### Step 2: bring the VIBE branch up to date

```sh
cd /mnt/data/ivf_proj/vibe
git fetch upstream && git switch add-qa-scann
git rebase upstream/main      # only needed if upstream has moved (it was 7ac76c1 on 2026-10-19)
rm -rf vibe/algorithms/qa_scann/__pycache__
cp /mnt/data/ivf_proj/QA-ScaNN/vibe/qa_scann/{config.yml,module.py,image.def} vibe/algorithms/qa_scann/
git status --short            # expect only: README.md and vibe/algorithms/qa_scann/
```

The README row in the Algorithms table (already added) should read
`| [QA-ScaNN](https://github.com/HNSWbeatsLSH/QA-ScaNN) | 0.1.1 |`, between
PyNNDescent and RoarGraph.

### Step 3: test the exact submitted files

```sh
# Apptainer image builds need a temp/build dir that is NOT setgid and group "storage"
# (most of /mnt/data/ivf_proj is); use the prepared one:
./install.sh --algorithm qa_scann --force --build-dir /mnt/data/ivf_proj/tmp/vibe-build
python3 run.py --dataset yi-128-ip --module qa_scann --runs 1 --parallelism 16 --force
```

Expected: 18 `qa-scann` builds, no errors, results in
`results/yi-128-ip/100/qa-scann/`, and recall@100 0.9 at about 18,000 QPS on
this machine. Nothing in `run.py` assumes a particular result-folder name, so
the old `qa-scann-data`/`qa-scann-queries` folders can stay for reference.

### Step 4: commit, push to the fork, open the PR

```sh
git add README.md vibe/algorithms/qa_scann
git -c user.name="Ashwin Padaki" -c user.email="apadaki@engineering.upenn.edu" \
    commit -m "Add QA-ScaNN (ScaNN with query-aware placement, OOD datasets)"
git push -u origin add-qa-scann
gh pr create --repo vector-index-bench/vibe --base main --head apadaki:add-qa-scann \
    --title "Add QA-ScaNN" --body-file pr_body.md
```

### Step 5: PR description (`pr_body.md`)

```markdown
Adds **QA-ScaNN** (https://github.com/HNSWbeatsLSH/QA-ScaNN, v0.1.1): ScaNN 1.4.2
with query-aware placement. Using the learning queries of the out-of-distribution
datasets, it stores extra copies of points in the leaves those queries probe
(a greedy that maximises (query, true neighbour) coverage per unit of added scan
cost, up to a target coverage `alpha` and a cap on mean copies per point). Search is
ScaNN's own.

- `vibe/algorithms/qa_scann/`: `image.def` (builds the patched ScaNN from source and
  installs qa_scann 0.1.1), `module.py` (`fit_ood` only, like RoarGraph: OOD datasets
  only), `config.yml` (one algorithm, 18 builds, at most 56 search settings per build).
- Dot-product ScaNN index (residual codes) for ip / normalized / cosine data.
- README row added.

Tested with VIBE's harness on all 8 OOD datasets (AMD EPYC 9575F, --runs 1); VIBE's
`scann` and `ivf(faiss)` were run alongside as references.

Resource notes, and questions for the next results update:
- Builds are single-threaded. Per-build peak memory: ≤ 20 GB on yi, llama, yandex,
  imagenet-align and laion; about 40 GB on hotpotqa; 52 GB on cqadupstack-lemur;
  128 GB on cqadupstack-muvera.
- On cqadupstack-muvera, the `dims_per_block = 1` configs are slow (on our machine
  up to 3.1 h to build and 4.4 h in total with one search pass; we estimate about
  10 h on Turin-class CPUs with 5 search runs, but possibly more than 24 h on the
  Xeon Gold 6230). The other
  configs give QA-ScaNN's best muvera results. Which hardware and per-job memory
  will the next update use, and would you prefer we drop `dims_per_block = 1` (it
  matters for the 128-d datasets)?
```

### Step 6: after opening

* If the maintainers ask for a smaller grid, cut `dims_per_block = 1` (cheapest
  on cqadupstack) or a `p` value. The ranks in section 5 show what each choice
  costs.
* If they rerun on Turin hardware, the timeout concern goes away. Their published
  ranks should then be comparable to the forecast in section 4.

---

## 8. Where everything is

| what | path |
|---|---|
| this report, ScaNN/QA-ScaNN explainer | `QA-ScaNN/docs/VIBE_RUN_AND_SUBMISSION.md`, `QA-ScaNN/docs/SCANN_AND_QA_SCANN.md` |
| run 3 (final): per-dataset tables, forecasts, logs, memory log | `/mnt/data/ivf_proj/tmp/vibe-run3/` (`summary/`, `logs/`, `README.txt`, `status.sh`, `forecast.py`, `published.json`) |
| VIBE results (run 3 QA-ScaNN, baselines, imagenet-align tests) | `/mnt/data/ivf_proj/vibe/results/<dataset>/100/` |
| earlier runs (superseded configs) | `/mnt/data/ivf_proj/tmp/vibe-run/` (run 1, `old_qa_results/` 18 GB), `/mnt/data/ivf_proj/tmp/vibe-run2/` (run 2) |
| imagenet-align tests (threshold sweep, `alpha` 0, dot product) | `/mnt/data/ivf_proj/tmp/vibe-test/` (`analyze.py`) |
| images | `/mnt/data/ivf_proj/vibe/images/` |
| Apptainer build dir (not setgid) | `/mnt/data/ivf_proj/tmp/vibe-build/` |

To regenerate a forecast, for example after rerunning a dataset, run
`/mnt/data/ivf_proj/env/bin/python /mnt/data/ivf_proj/tmp/vibe-run3/forecast.py <dataset>`.
It writes `summary/<dataset>_forecast.txt`.
