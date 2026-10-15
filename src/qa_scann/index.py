"""QA-ScaNN: ScaNN with query-aware placement.

ScaNN stores each point in one leaf of its k-means tree. QA-ScaNN keeps that
index -- tree, leaf assignments, AH codebook, codes and search -- and adds extra
copies of points in further leaves, chosen from a sample of the query
distribution (the learning queries):

  0. the tree is trained by ScaNN's k-means on the dataset points (ScaNN's
     default) or on the learning queries (centroids="queries"); either way
     ScaNN assigns each point to its nearest centre and trains its codebook
     on that partition;
  1. route every learning query to its top-p leaves, as the index routes;
  2. a demand pair (query, one of its true top-k neighbours) is served when
     the neighbour is stored in one of those p leaves; the leaf a point already
     sits in serves some pairs for free;
  3. a greedy min-cost partial cover adds (point, leaf) copies, each costing the
     number of learning queries that probe the leaf, until a fraction alpha of
     all demand pairs is served;
  4. the copies are stored as extra leaf memberships. With residual AH codes
     (dot product) each copy is encoded against its own leaf centre; with
     non-residual codes (squared L2) every copy shares the point's code.

Search is ScaNN's own: probe leaves, score with AH lookup tables, re-rank a
shortlist exactly. The patched ScaNN merges a point's copies in the shortlist,
so results hold distinct points.
"""
import gc
import json
import os
import shutil
import tempfile

import numpy as np

from . import _core
from . import scann_io as io

__all__ = ["QAScaNN"]

_META = "qa_scann.json"
_CHUNK = 16384


def _as_f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _top_partitions(q, cen, distance, p):
    """(m, p) int32: each query's p best leaves, best first."""
    cen = _as_f32(cen)
    half = 0.5 * np.einsum("ij,ij->i", cen, cen) if distance == "squared_l2" else None
    out = np.empty((len(q), p), np.int32)
    for a in range(0, len(q), _CHUNK):
        s = _as_f32(q[a:a + _CHUNK]) @ cen.T
        if half is not None:
            s -= half                          # argmax q.c - |c|^2/2 == argmin |q - c|^2
        part = np.argpartition(-s, p - 1, axis=1)[:, :p] if p < cen.shape[0] else \
            np.tile(np.arange(cen.shape[0]), (len(s), 1))
        sc = np.take_along_axis(s, part, 1)
        o = np.lexsort((part, -sc), axis=1)   # by score, ties by leaf id
        out[a:a + _CHUNK] = np.take_along_axis(part, o, 1)
    return out


def _exact_neighbors(q, X, distance, k):
    """(m, k) int32 exact top-k of q among X (brute force, blocked)."""
    X = _as_f32(X)
    half = 0.5 * np.einsum("ij,ij->i", X, X) if distance == "squared_l2" else None
    out = np.empty((len(q), k), np.int32)
    for a in range(0, len(q), 1024):
        s = _as_f32(q[a:a + 1024]) @ X.T
        if half is not None:
            s -= half
        part = np.argpartition(-s, k - 1, axis=1)[:, :k]
        o = np.argsort(-np.take_along_axis(s, part, 1), axis=1, kind="stable")
        out[a:a + 1024] = np.take_along_axis(part, o, 1)
    return out


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class QAScaNN:
    """ScaNN with query-aware placement.

    Parameters
    ----------
    distance : "squared_l2" or "dot_product"
        ScaNN's distance. For cosine similarity, normalise the vectors and use
        either (on unit vectors both rank identically).
    p : int
        Leaves per learning query the placement is optimised for.
    alpha : float
        Fraction of the (learning query, true neighbour) pairs the placement
        must serve; larger alpha stores more copies.
    centroids : "data" or "queries"
        What ScaNN's k-means tree is trained on: a sample of the dataset
        points (ScaNN's default) or of the learning queries. Either way ScaNN
        then assigns every point to its nearest centre and trains its AH
        codebook on that partition, so codes and centres stay matched.
    num_leaves : int or None
        Leaves of ScaNN's k-means tree; None = floor(sqrt(n)).
    dims_per_block, anisotropic_quantization_threshold, training_sample_size
        ScaNN's AH and tree settings (defaults: 2, 0.2, 250,000).
    demand_k : int
        Neighbours per learning query that count as demand (recall@k target).
    max_mean_copies : float or None
        Cap on the mean copies per point (home copy included): the placement
        stops at whichever comes first, alpha coverage or (cap - 1) * n extra
        copies. None = no cap.
    training_threads : int
        Threads for ScaNN's tree and codebook training (the placement and the
        encoder are single-threaded).
    """

    def __init__(self, distance="squared_l2", p=4, alpha=0.9, centroids="data", num_leaves=None,
                 dims_per_block=2, anisotropic_quantization_threshold=0.2,
                 training_sample_size=250_000, demand_k=100, training_threads=1, max_mean_copies=None):
        if distance not in ("squared_l2", "dot_product"):
            raise ValueError("distance must be 'squared_l2' or 'dot_product'")
        if centroids not in ("data", "queries"):
            raise ValueError("centroids must be 'data' or 'queries'")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if p < 1:
            raise ValueError("p must be >= 1")
        if max_mean_copies is not None and not max_mean_copies >= 1:
            raise ValueError("max_mean_copies must be >= 1 (or None)")
        self.distance = distance
        self.p = int(p)
        self.alpha = float(alpha)
        self.centroids = centroids
        self.num_leaves = num_leaves
        self.dims_per_block = int(dims_per_block)
        self.anisotropic_quantization_threshold = float(anisotropic_quantization_threshold)
        self.training_sample_size = int(training_sample_size)
        self.demand_k = int(demand_k)
        self.training_threads = int(training_threads)
        self.max_mean_copies = None if max_mean_copies is None else float(max_mean_copies)
        self.searcher = None
        self.stats = {}
        self._dir = None
        self._owns_dir = False

    # ------------------------------------------------------------ building

    def fit(self, X, learn, learn_neighbors=None):
        """Build ScaNN on X, then place copies for the learning queries.

        X (n, d) float32; learn (m, d) a sample of the query distribution;
        learn_neighbors (m, >= demand_k) its exact neighbours in X by
        `distance` (computed by brute force when None).
        """
        scann = io.require_patched_scann()
        X = _as_f32(X)
        n = X.shape[0]
        nl = int(self.num_leaves or max(int(np.floor(np.sqrt(n))), 1))
        work = tempfile.mkdtemp(prefix="qa_scann_")
        try:
            if self.centroids == "data":
                base = self._builder(scann, X, nl).build()
            else:
                base = self._build_on_query_centroids(scann, X, learn, nl, work)
            base.serialize(work, relative_path=True)
            del base
            gc.collect()
            self._place(work, work, X, learn, learn_neighbors, self.centroids)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        self._set_dir(work, owns=True)
        self._load()
        return self

    def _builder(self, scann, X, nl):
        """ScaNN's builder for X: k-means tree, AH scoring, exact re-ranking."""
        b = scann.scann_ops_pybind.builder(X, 10, self.distance)
        b.set_n_training_threads(self.training_threads)
        return (b.tree(num_leaves=nl, num_leaves_to_search=min(3200, nl),
                       training_sample_size=min(self.training_sample_size, X.shape[0]))
                 .score_ah(self.dims_per_block,
                           anisotropic_quantization_threshold=self.anisotropic_quantization_threshold)
                 .reorder(20000))

    def _build_on_query_centroids(self, scann, X, learn, nl, work):
        """ScaNN's index of X with its k-means tree trained on the learning queries.

        ScaNN's build in its stages: (1) its k-means trainer runs on the
        learning queries (a tree-only index of them); (2) that serialized tree
        is handed to ScaNN's factory together with X and the index config of
        X, as a pre-trained partitioner; the factory then assigns every point
        of X to its nearest centre and trains the AH codebook and codes on that
        partition, as it would after training the tree itself.
        """
        from google.protobuf import text_format
        from scann.proto import scann_pb2
        from scann.scann_ops.py import scann_ops_pybind
        L = _as_f32(learn)
        if len(L) < nl:
            raise ValueError("centroids='queries' needs at least num_leaves (%d) learning "
                             "queries, got %d" % (nl, len(L)))
        qb = scann.scann_ops_pybind.builder(L, 10, self.distance)
        qb.set_n_training_threads(self.training_threads)
        qtree = (qb.tree(num_leaves=nl, num_leaves_to_search=min(3200, nl),
                         training_sample_size=min(self.training_sample_size, len(L)))
                   .score_brute_force()
                   .build())
        qdir = tempfile.mkdtemp(prefix="qa_scann_qtree_")
        try:
            qtree.serialize(qdir, relative_path=True)
            del qtree
            shutil.copyfile(os.path.join(qdir, "serialized_partitioner.pb"),
                            os.path.join(work, "serialized_partitioner.pb"))
        finally:
            shutil.rmtree(qdir, ignore_errors=True)
        cfg = scann_pb2.ScannConfig()
        text_format.Parse(self._builder(scann, X, nl).create_config(), cfg)
        with open(os.path.join(work, "scann_config.pb"), "wb") as f:
            f.write(cfg.SerializeToString())
        np.save(os.path.join(work, "dataset.npy"), X)
        assets = ('assets {\n  asset_type: PARTITIONER\n  asset_path: "serialized_partitioner.pb"\n}\n'
                  'assets {\n  asset_type: DATASET_NPY\n  asset_path: "dataset.npy"\n}\n')
        gc.collect()
        return scann_ops_pybind.ScannSearcher(scann_ops_pybind.scann_pybind.ScannNumpy(work, assets), None)

    def fit_from_scann(self, scann_dir, learn, learn_neighbors=None, X=None, out_dir=None):
        """Place copies on an existing serialized ScaNN index (one leaf per point).

        The index is not modified; the QA-ScaNN index is written to out_dir
        (a temporary directory when None). X defaults to the index's own
        dataset.npy. The tree is the given index's own, so `centroids` does
        not apply here.
        """
        io.require_patched_scann()
        if X is None:
            X = np.load(os.path.join(scann_dir, "dataset.npy"), mmap_mode="r")
        X = _as_f32(X)
        cfg = io.read_config(scann_dir)
        self.distance = "dot_product" if "Dot" in cfg.distance_measure.distance_measure else "squared_l2"
        owns = out_dir is None
        out = out_dir or tempfile.mkdtemp(prefix="qa_scann_")
        os.makedirs(out, exist_ok=True)
        try:
            self._place(scann_dir, out, X, learn, learn_neighbors, "given index")
        except BaseException:
            if owns:
                shutil.rmtree(out, ignore_errors=True)
            raise
        self._set_dir(out, owns=owns)
        self._load()
        return self

    def _place(self, base_dir, out_dir, X, learn, learn_neighbors, centroids):
        cfg = io.read_config(base_dir)
        route_dist = io.route_distance(cfg)
        residual = io.is_residual(cfg)
        cen = io.read_centers(base_dir)
        nl, n = cen.shape[0], X.shape[0]
        home = np.load(os.path.join(base_dir, "datapoint_to_token.npy")).astype(np.int32).ravel()
        if home.shape != (n,):
            raise ValueError("the base index must store each point in exactly one leaf")
        p = min(self.p, nl)

        route = _top_partitions(learn, cen, route_dist, p)
        if learn_neighbors is None:
            owner = _exact_neighbors(learn, X, self.distance, self.demand_k)
        else:
            nb = np.asarray(learn_neighbors)
            k = min(self.demand_k, nb.shape[1])
            owner = np.ascontiguousarray(nb[:, :k], dtype=np.int32)
        m, k = owner.shape
        keys, gains, base_cov = _core.greedy(owner, route, home, n, nl)
        del owner, route

        # the shortest prefix of the greedy order that serves alpha of the demand
        need = self.alpha * m * k
        cum = base_cov + np.concatenate([[0], np.cumsum(gains)])
        hit = np.flatnonzero(cum >= need - 1e-9)
        steps = int(hit[0]) if len(hit) else len(keys)
        capped = False
        if self.max_mean_copies is not None and steps > int((self.max_mean_copies - 1) * n):
            steps, capped = int((self.max_mean_copies - 1) * n), True
        keys = keys[:steps]
        pt, lst = keys // nl, keys % nl

        # CSR by point: home leaf first, then the copies in greedy order
        rows = np.concatenate([np.arange(n, dtype=np.int64), pt])
        lists = np.concatenate([home.astype(np.int64), lst])
        extra = np.concatenate([np.zeros(n, bool), np.ones(len(pt), bool)])
        o = np.lexsort((np.arange(len(rows)), extra, rows))
        rows, lists, extra = rows[o], lists[o], extra[o]
        cnt = np.bincount(rows, minlength=n)
        off = np.zeros(n + 1, np.int64)
        np.cumsum(cnt, out=off[1:])

        assets = []
        for t, path in io.read_assets(base_dir):
            if t == "TOKENIZATION_NPY" or (residual and t == "AH_DATASET_NPY"):
                continue
            if t in io.MULTI_ASSETS or t in ("AH_DATASET_SOAR_NPY",):
                raise ValueError("the base index already has spilled assets (%s)" % t)
            name = os.path.basename(path)
            if os.path.abspath(os.path.dirname(path)) != os.path.abspath(out_dir):
                _link_or_copy(path, os.path.join(out_dir, name))
            assets.append((t, name))
        np.save(os.path.join(out_dir, "datapoint_to_token_multi.npy"), lists.astype(np.int32))
        np.save(os.path.join(out_dir, "datapoint_to_token_offsets.npy"), off)
        assets += [("TOKENIZATION_MULTI_NPY", "datapoint_to_token_multi.npy"),
                   ("TOKENIZATION_MULTI_OFFSETS_NPY", "datapoint_to_token_offsets.npy")]
        if residual:
            B = io.read_codebook(base_dir)
            prim = np.load(os.path.join(base_dir, "hashed_dataset.npy"), mmap_mode="r")
            H = np.empty((len(rows), prim.shape[1]), np.uint8)
            hi = np.flatnonzero(~extra)
            H[hi] = prim[rows[hi]]
            ei = np.flatnonzero(extra)
            t = io.noise_shaping_threshold(cfg)
            for a in range(0, len(ei), 1 << 20):
                e = ei[a:a + (1 << 20)]
                H[e] = _core.ah_encode_residual(X, cen, rows[e], lists[e], B, t)
            del prim
            np.save(os.path.join(out_dir, "hashed_dataset_multi.npy"), H)
            del H
            assets.append(("AH_DATASET_MULTI_NPY", "hashed_dataset_multi.npy"))
        if os.path.abspath(base_dir) != os.path.abspath(out_dir):
            shutil.copyfile(os.path.join(base_dir, "scann_config.pb"),
                            os.path.join(out_dir, "scann_config.pb"))
        else:
            for f in ("datapoint_to_token.npy",) + (("hashed_dataset.npy",) if residual else ()):
                if os.path.exists(os.path.join(out_dir, f)):
                    os.remove(os.path.join(out_dir, f))
        io.write_assets(out_dir, assets)

        self.num_leaves = nl
        self.stats = {
            "n": int(n), "num_leaves": int(nl), "centroids": centroids, "p": int(p),
            "alpha": self.alpha, "max_mean_copies": self.max_mean_copies, "capped": capped,
            "learning_queries": int(m), "demand_k": int(k),
            "greedy_steps": int(steps), "full_cover_steps": int(len(gains)),
            "coverage": float(cum[steps] / (m * k)) if m * k else 1.0,
            "home_coverage": float(base_cov / (m * k)) if m * k else 1.0,
            "copies_per_point": float(len(rows) / n), "max_copies": int(cnt.max()),
            "route_distance": route_dist,
            "codes": "residual, one per copy" if residual else "non-residual, shared by copies",
        }
        params = {k2: getattr(self, k2) for k2 in (
            "distance", "p", "alpha", "centroids", "num_leaves", "dims_per_block",
            "anisotropic_quantization_threshold", "training_sample_size", "demand_k", "max_mean_copies")}
        with open(os.path.join(out_dir, _META), "w") as f:
            json.dump({"params": params, "stats": self.stats, "assets": assets}, f, indent=1)

    # ------------------------------------------------------------ loading

    def _set_dir(self, d, owns):
        if self._owns_dir and self._dir and self._dir != d:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir, self._owns_dir = d, owns

    def _load(self):
        """Load the patched ScaNN searcher from self._dir."""
        io.require_patched_scann()
        from scann.scann_ops.py import scann_ops_pybind
        with open(os.path.join(self._dir, _META)) as f:
            assets = json.load(f)["assets"]
        # absolute paths, so the directory can be moved between save and load
        txt = "".join('assets {\n  asset_type: %s\n  asset_path: "%s"\n}\n'
                      % (t, os.path.abspath(os.path.join(self._dir, name)))
                      for t, name in assets)
        self.searcher = None
        gc.collect()
        self.searcher = scann_ops_pybind.ScannSearcher(
            scann_ops_pybind.scann_pybind.ScannNumpy(self._dir, txt), None)
        _malloc_trim()

    def save(self, path):
        """Write the index to directory `path` (portable: relative asset paths)."""
        if self._dir is None:
            raise RuntimeError("nothing to save; call fit() first")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(self._dir, _META)) as f:
            meta = json.load(f)
        for _, name in meta["assets"]:
            shutil.copyfile(os.path.join(self._dir, name), os.path.join(path, name))
        for f in ("scann_config.pb", "scann_assets.pbtxt", _META):
            shutil.copyfile(os.path.join(self._dir, f), os.path.join(path, f))

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, _META)) as f:
            meta = json.load(f)
        self = cls(**{k: v for k, v in meta["params"].items()})
        self.stats = meta["stats"]
        self._set_dir(os.path.abspath(path), owns=False)
        self._load()
        return self

    def close(self):
        self.searcher = None
        if self._owns_dir and self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir, self._owns_dir = None, False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------ search

    def _args(self, k, leaves_to_search, reorder):
        L = int(min(max(int(leaves_to_search), 1), self.num_leaves))
        return int(k), int(max(int(reorder), int(k))), L

    def search(self, q, k, leaves_to_search, reorder):
        """Top-k (ids, distances) for one query.

        leaves_to_search: leaves probed; reorder: AH shortlist re-ranked
        exactly. Fewer than k results come back only if the probed leaves
        hold fewer than k points.
        """
        k, ro, L = self._args(k, leaves_to_search, reorder)
        idx, dist = self.searcher.search(_as_f32(q), k, ro, L)
        ok = ~np.isnan(dist)
        return np.asarray(idx)[ok], np.asarray(dist)[ok]

    def search_batched(self, Q, k, leaves_to_search, reorder, parallel=False):
        """(ids, distances), each (nq, k); missing results are id -1, NaN distance."""
        k, ro, L = self._args(k, leaves_to_search, reorder)
        fn = self.searcher.search_batched_parallel if parallel else self.searcher.search_batched
        idx, dist = fn(_as_f32(Q), k, ro, L)
        idx = np.array(idx, dtype=np.int64)
        dist = np.asarray(dist)
        idx[np.isnan(dist)] = -1
        return idx, dist
