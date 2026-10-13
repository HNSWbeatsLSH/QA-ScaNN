#!/usr/bin/env python3
"""Check that QA-ScaNN's staged ScaNN build reproduces ScaNN's normal build.

centroids="queries" builds ScaNN in stages: a trained k-means tree is handed to
ScaNN's index factory as a pre-trained partitioner, and the factory assigns
the points and trains the AH codebook. Fed ScaNN's own data-trained tree, the
staged build must give the normal build byte for byte; then the only thing
centroids="queries" changes is what the k-means tree is trained on.

    python tools/check_staged_build.py DATA [--distance squared_l2|dot_product]

DATA is a .npy (n, d) float32 file or a VIBE .hdf5 file (its `train` set).
Needs the patched ScaNN. Exit status 0 when every index file is identical.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

from qa_scann import QAScaNN
from qa_scann.scann_io import require_patched_scann
from _common import load_vectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--distance", default="squared_l2", choices=["squared_l2", "dot_product"])
    ap.add_argument("--normalize", action="store_true", help="normalise rows (cosine data)")
    a = ap.parse_args()
    scann = require_patched_scann()
    from google.protobuf import text_format
    from scann.proto import scann_pb2
    from scann.scann_ops.py import scann_ops_pybind

    X = load_vectors(a.data, "train", a.normalize)
    q = QAScaNN(distance=a.distance)
    nl = max(int(np.floor(np.sqrt(len(X)))), 1)
    normal, staged, out = (tempfile.mkdtemp() for _ in range(3))
    try:
        q._builder(scann, X, nl).build().serialize(normal, relative_path=True)
        shutil.copyfile(os.path.join(normal, "serialized_partitioner.pb"),
                        os.path.join(staged, "serialized_partitioner.pb"))
        cfg = scann_pb2.ScannConfig()
        text_format.Parse(q._builder(scann, X, nl).create_config(), cfg)
        with open(os.path.join(staged, "scann_config.pb"), "wb") as f:
            f.write(cfg.SerializeToString())
        np.save(os.path.join(staged, "dataset.npy"), X)
        assets = ('assets {\n asset_type: PARTITIONER\n asset_path: "serialized_partitioner.pb"\n}\n'
                  'assets {\n asset_type: DATASET_NPY\n asset_path: "dataset.npy"\n}\n')
        s = scann_ops_pybind.ScannSearcher(scann_ops_pybind.scann_pybind.ScannNumpy(staged, assets), None)
        s.serialize(out, relative_path=True)
        ok = True
        for f in sorted(os.listdir(normal)):
            x, y = os.path.join(normal, f), os.path.join(out, f)
            if not os.path.exists(y):
                same = False
            elif f.endswith(".npy"):
                u, v = np.load(x), np.load(y)
                same = u.shape == v.shape and np.array_equal(u, v)
            else:
                same = open(x, "rb").read() == open(y, "rb").read()
            ok &= same
            print("  %-28s %s" % (f, "identical" if same else "DIFFERENT"))
        print("staged build == normal build" if ok else "staged build differs")
        return 0 if ok else 1
    finally:
        for d in (normal, staged, out):
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
