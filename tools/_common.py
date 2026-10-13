"""Helpers shared by the scripts in tools/ (not part of the package)."""
import numpy as np


def load_vectors(path, key="train", normalize=False):
    """float32 rows of a .npy file, or of `key` in a VIBE .hdf5 file."""
    if path.endswith((".hdf5", ".h5")):
        import h5py
        with h5py.File(path, "r") as f:
            X = np.asarray(f[key][:])
    else:
        X = np.load(path)
    X = np.ascontiguousarray(X, dtype=np.float32)
    if normalize:
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        X = np.ascontiguousarray(X / nrm, dtype=np.float32)
    return X
