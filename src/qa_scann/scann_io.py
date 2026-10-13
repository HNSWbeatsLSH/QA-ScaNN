"""Readers and writers for a serialized ScaNN index directory."""
import os

import numpy as np

MULTI_ASSETS = ("TOKENIZATION_MULTI_NPY", "TOKENIZATION_MULTI_OFFSETS_NPY", "AH_DATASET_MULTI_NPY")


def require_patched_scann():
    """Import scann and check it is the multi-token build this package needs."""
    try:
        import scann
        from scann.scann_ops import scann_assets_pb2
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "qa_scann needs the patched ScaNN (scann 1.4.2+qa1); build it with "
            "scripts/build_scann.sh") from e
    names = scann_assets_pb2.ScannAsset.AssetType.keys()
    if not all(a in names for a in MULTI_ASSETS):
        raise ImportError(
            "the installed scann (%s) cannot load indexes with more than two "
            "partitions per point; install the patched build from "
            "scripts/build_scann.sh" % getattr(scann, "__file__", "?"))
    return scann


def read_config(index_dir):
    from scann.proto import scann_pb2
    cfg = scann_pb2.ScannConfig()
    with open(os.path.join(index_dir, "scann_config.pb"), "rb") as f:
        cfg.ParseFromString(f.read())
    return cfg


def route_distance(cfg):
    """'dot_product' or 'squared_l2': how the index routes a query to leaves."""
    name = cfg.partitioning.query_tokenization_distance_override.distance_measure
    if not name:
        name = cfg.partitioning.partitioning_distance.distance_measure
    if not name:
        name = cfg.distance_measure.distance_measure
    return "dot_product" if "Dot" in name else "squared_l2"


def is_residual(cfg):
    return bool(cfg.hash.asymmetric_hash.use_residual_quantization)


def noise_shaping_threshold(cfg):
    ah = cfg.hash.asymmetric_hash
    return float(ah.noise_shaping_threshold) if ah.HasField("noise_shaping_threshold") else float("nan")


def read_centers(index_dir):
    """Leaf centres of a one-level k-means tree, (nlist, d) float32, by leaf id."""
    from scann.partitioning import partitioner_pb2
    p = partitioner_pb2.SerializedPartitioner()
    with open(os.path.join(index_dir, "serialized_partitioner.pb"), "rb") as f:
        p.ParseFromString(f.read())
    root = p.kmeans.kmeans_tree.root
    if any(len(c.children) for c in root.children):
        raise NotImplementedError("only one-level k-means trees are supported")
    cen = np.array([list(c.dimension) if len(c.dimension) else list(c.float_dimension)
                    for c in root.centers], dtype=np.float64)
    leaf = np.array([c.leaf_id for c in root.children], dtype=np.int64)
    if len(leaf) != len(cen) or not np.array_equal(leaf, np.arange(len(cen))):
        raise ValueError("unexpected leaf ids in the partitioner")
    return np.ascontiguousarray(cen, dtype=np.float32)


def read_codebook(index_dir):
    """AH codebook, (blocks, codewords, dims per block) float64."""
    from scann.proto import centers_pb2
    cb = centers_pb2.CentersForAllSubspaces()
    with open(os.path.join(index_dir, "ah_codebook.pb"), "rb") as f:
        cb.ParseFromString(f.read())
    blocks = [[list(c.feature_value_float) for c in s.center] for s in cb.subspace_centers]
    if len({(len(b), len(b[0])) for b in blocks}) != 1:
        raise NotImplementedError("AH blocks of unequal size are not supported")
    return np.array(blocks, dtype=np.float64)


def read_assets(index_dir):
    """[(asset_type, path)] of scann_assets.pbtxt, paths resolved against index_dir."""
    from google.protobuf import text_format
    from scann.scann_ops import scann_assets_pb2
    assets = scann_assets_pb2.ScannAssets()
    with open(os.path.join(index_dir, "scann_assets.pbtxt")) as f:
        text_format.Parse(f.read(), assets)
    out = []
    for a in assets.assets:
        t = scann_assets_pb2.ScannAsset.AssetType.Name(a.asset_type)
        path = a.asset_path if os.path.isabs(a.asset_path) else os.path.join(index_dir, a.asset_path)
        out.append((t, path))
    return out


def write_assets(index_dir, assets):
    """scann_assets.pbtxt listing `assets` [(asset_type, file name in index_dir)]."""
    with open(os.path.join(index_dir, "scann_assets.pbtxt"), "w") as f:
        for t, name in assets:
            f.write('assets {\n  asset_type: %s\n  asset_path: "%s"\n}\n' % (t, name))
