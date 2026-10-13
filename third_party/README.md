# third_party

## scann/

This is ScaNN 1.4.2 with the multi-token patch applied. It is the `scann/`
directory of [google-research](https://github.com/google-research/google-research)
at commit `634a79fb5b4b23237a81eac028d27b5d34a5dba4` (2025-08-28). The copy is
licensed under the Apache License 2.0; see `scann/LICENSE`, which is the
google-research repository's license file.

`scripts/build_scann.sh` builds the wheel from this copy, so building QA-ScaNN
does not need the upstream repository.

### What the patch changes

`scann-multi-token.patch` is the complete diff against the upstream commit:
10 files, +270/−16. It changes four things:

- **More than two leaves per point.** New index assets
  `TOKENIZATION_MULTI_NPY` and `TOKENIZATION_MULTI_OFFSETS_NPY` (a CSR list
  of leaves per point, home leaf first), plus an optional
  `AH_DATASET_MULTI_NPY` (one AH code per point and leaf). These let a point
  sit in any number of leaves. Defined in `scann_ops/scann_assets.proto`,
  loaded in `scann_ops/cc/scann.cc`.
- **Searching such indexes.** The tree-AH searchers accept a point in more
  than two leaves (`tree_x_hybrid/*`). The over-retrieval factor may exceed
  2 (`base/internal/tree_x_hybrid_factory.cc`). Duplicate copies in a
  shortlist are merged by their mean score, as ScaNN's SOAR merges its two
  copies (`tree_x_hybrid/internal/utils.cc`). The environment variable
  `SCANN_MULTI_SPILL_DEDUP=min|max` selects min or max instead.
- **Codes without the new asset.** Without `AH_DATASET_MULTI_NPY`, the
  copies of a point share its single row of `AH_DATASET_NPY`. This is
  correct for non-residual (squared-L2) codes.
- **Version.** The version is `1.4.2+qa1` (`pyproject.toml`).

On indexes without the new assets, the patched build returns exactly what
stock ScaNN 1.4.2 returns.

### Regenerating the copy from upstream

```bash
git init google-research && cd google-research
git remote add origin https://github.com/google-research/google-research.git
git sparse-checkout set scann
git fetch --depth 1 --filter=blob:none origin 634a79fb5b4b23237a81eac028d27b5d34a5dba4
git checkout FETCH_HEAD
git apply ../third_party/scann-multi-token.patch
# then: third_party/scann = google-research/scann, plus google-research/LICENSE
```
