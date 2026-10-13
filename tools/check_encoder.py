#!/usr/bin/env python3
"""Compare QA-ScaNN's AH encoder with ScaNN's own codes on a residual index.

QA-ScaNN encodes the extra copies of points itself (residual codes, one per
copy). This re-encodes every point against its own leaf with that encoder and
compares the result with the codes ScaNN stored for the index.

    python tools/check_encoder.py INDEX_DIR

INDEX_DIR is a serialized ScaNN dot-product index (residual AH codes), e.g.
the base index QA-ScaNN builds for inner-product data. Prints how many points
get exactly ScaNN's code, and how many blocks differ elsewhere.
"""
import os
import sys

import numpy as np

from qa_scann import _core
from qa_scann import scann_io as io


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    io.require_patched_scann()
    cfg = io.read_config(d)
    if not io.is_residual(cfg):
        print("not a residual-code index: copies share ScaNN's code, nothing to check")
        return 0
    X = np.load(os.path.join(d, "dataset.npy"), mmap_mode="r")
    cen = io.read_centers(d)
    B = io.read_codebook(d)
    t = io.noise_shaping_threshold(cfg)
    home = np.load(os.path.join(d, "datapoint_to_token.npy")).astype(np.int64).ravel()
    codes = np.load(os.path.join(d, "hashed_dataset.npy"), mmap_mode="r")
    n = len(home)
    same, blocks = 0, []
    for a in range(0, n, 50_000):
        r = np.arange(a, min(n, a + 50_000))
        c = _core.ah_encode_residual(np.ascontiguousarray(X[r], dtype=np.float32), cen,
                                     np.arange(len(r), dtype=np.int64), home[r], B, t)
        diff = (c != codes[r]).sum(1)
        same += int((diff == 0).sum())
        blocks += diff[diff > 0].tolist()
    print("%d of %d points encode exactly as ScaNN encoded them" % (same, n))
    if blocks:
        print("the other %d differ in %s of %d blocks" % (len(blocks), sorted(set(blocks)), codes.shape[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
