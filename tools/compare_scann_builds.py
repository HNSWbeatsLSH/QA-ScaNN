#!/usr/bin/env python3
"""Check that two ScaNN installs (e.g. PyPI scann==1.4.2 and the patched
1.4.2+qa1) return the same results on the same serialized ScaNN index.

Run `search` once with each install (each in its own environment), then `diff`:

    python tools/compare_scann_builds.py search INDEX_DIR QUERIES OUT.npz [--normalize]
    python tools/compare_scann_builds.py diff A.npz B.npz

INDEX_DIR is a directory written by ScaNN's `serialize`; QUERIES a .npy file
or a VIBE .hdf5 file (its `test` set). Every (leaves, reorder) setting below is
searched for the top 100. `diff` exits 0 when ids and distances are identical.
"""
import argparse
import sys

import numpy as np

SETTINGS = ((1, 200), (5, 1000), (20, 5000), (100, 20000))


def search(index_dir, queries, out, normalize):
    import scann
    from _common import load_vectors
    s = scann.scann_ops_pybind.load_searcher(index_dir)
    q = load_vectors(queries, "test", normalize)
    res = {}
    for L, r in SETTINGS:
        I, D = s.search_batched(q, 100, r, L)
        res["L%d_r%d_ids" % (L, r)] = np.asarray(I)
        res["L%d_r%d_dists" % (L, r)] = np.asarray(D)
    np.savez(out, **res)
    print("searched with", scann.__file__, "->", out)
    return 0


def diff(a, b):
    A, B = np.load(a), np.load(b)
    ok = sorted(A.files) == sorted(B.files)
    for k in sorted(A.files):
        same = k in B.files and np.array_equal(A[k], B[k], equal_nan=True)
        ok &= same
        print("  %-18s %s" % (k, "identical" if same else "DIFFERENT"))
    print("identical results" if ok else "results differ")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("index_dir")
    s.add_argument("queries")
    s.add_argument("out")
    s.add_argument("--normalize", action="store_true", help="normalise queries (cosine data)")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "search":
        return search(a.index_dir, a.queries, a.out, a.normalize)
    return diff(a.a, a.b)


if __name__ == "__main__":
    sys.exit(main())
