#!/usr/bin/env python3
"""Summarise QA-ScaNN results written by VIBE: best QPS at recall targets.

    python tools/vibe_summary.py VIBE_DIR DATASET [DATASET ...] [--targets 0.9 0.95]

VIBE_DIR is a VIBE checkout after `python3 run.py --dataset DATASET --algorithm
qa-scann` (results in results/DATASET/100/qa-scann/, ground truth in
data/DATASET.hdf5). For each centroid source it prints the highest QPS that
reaches each recall@100 target and the build that reached it, plus build times
and copies per point.
"""
import argparse
import glob
import os
import sys

import numpy as np


def summarise(vibe_dir, ds, targets):
    import h5py
    with h5py.File(os.path.join(vibe_dir, "data", ds + ".hdf5"), "r") as f:
        gt = np.asarray(f["neighbors"][:, :100])
    best, builds = {}, {}
    for fn in sorted(glob.glob(os.path.join(vibe_dir, "results", ds, "100", "qa-scann", "*.hdf5"))):
        if not os.path.exists(fn + ".complete"):
            continue
        with h5py.File(fn, "r") as h:
            for key in h.keys():
                g = h[key]
                a = dict(g.attrs)
                name = a["name"]
                src = name.split("centroids=")[1].split(",")[0] if "centroids=" in name else "data"
                cell = name.split(", leaves")[0].split("(")[1]
                I = np.asarray(g["neighbors"])
                rec = float(np.mean([len(np.intersect1d(i[i >= 0], t)) for i, t in zip(I, gt)]) / 100)
                qps = 1.0 / a["best_search_time"]
                builds[cell] = (src, a["build_time"], a["copies_per_point"])
                for t in targets:
                    if rec >= t and qps > best.get((src, t), (0.0, ""))[0]:
                        best[(src, t)] = (qps, "%s, leaves=%s, reorder=%s" % (cell, *key.split("_")))
    print("== %s" % ds)
    for src in sorted({v[0] for v in builds.values()}):
        b = [v for v in builds.values() if v[0] == src]
        print("  centroids=%s: %d builds, build %.0f-%.0f s, %.2f-%.2f copies/point" % (
            src, len(b), min(v[1] for v in b), max(v[1] for v in b),
            min(v[2] for v in b), max(v[2] for v in b)))
        for t in targets:
            r = best.get((src, t))
            print("    recall %.2f: %s" % (t, "%.0f QPS (%s)" % r if r else "not reached"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vibe_dir")
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.9, 0.95])
    a = ap.parse_args()
    for ds in a.datasets:
        summarise(a.vibe_dir, ds, a.targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
