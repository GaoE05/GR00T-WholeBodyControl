"""CPU contracts for a homogeneous SoftSONIC motion batch, before FK workers.

SoftMimic stores zero-wrench records with the same augmented schema as forced
records. Keep that invariant here instead of inferring capabilities in a worker.
"""
from queue import Empty

import numpy as np


def motion_schema(record, label="motion"):
    aug = "pose_aa_aug" in record
    if aug != ("root_trans_aug" in record):
        raise ValueError(f"{label}: pose_aa_aug and root_trans_aug must appear together")
    field = "softsonic" in record
    if not (aug or field):
        return False, False
    pose = np.asarray(record["pose_aa"])
    root = np.asarray(record["root_trans_offset"])
    n = len(pose)
    if n < 2 or pose.ndim != 3 or pose.shape[-1] != 3 or root.shape != (n, 3):
        raise ValueError(f"{label}: invalid reference pose/root dimensions or length")
    arrays = {"pose_aa": pose, "root_trans_offset": root}
    fps = float(record.get("fps", 30.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{label}: fps must be finite and positive")
    if aug:
        for key, shape in (("pose_aa_aug", pose.shape), ("root_trans_aug", root.shape)):
            arrays[key] = np.asarray(record[key])
            if arrays[key].shape != shape:
                raise ValueError(f"{label}: {key} shape differs from reference")
    if field:
        ss = arrays["softsonic"] = np.asarray(record["softsonic"])
        if ss.shape != (n, 15):
            raise ValueError(f"{label}: softsonic must have shape ({n}, 15)")
        if np.any(ss[:, 7:9] < 0):
            raise ValueError(f"{label}: negative field stiffness")
        active = (ss[:, 7:9] > 0).any(axis=1)
        ids = ss[active, 6]
        # Schema v1 uses the five names in SOFTSONIC_FORCE_BODIES, not MJCF ids.
        if np.any((ids < 0) | (ids >= 5) | (ids != np.floor(ids))):
            raise ValueError(f"{label}: invalid active canonical force body id")
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: {key} contains non-finite values")
    return aug, field


def inspect_motion_batch(records, cfg, is_evaluation, load_file):
    """Return one immutable (aug, field) capability tuple for this load.

    Read each distinct selected file once, retaining only its schema. Do not
    preload an entire large SONIC library or duplicate raw data in memory.
    Ordinary SONIC motions keep their existing augmentation behavior.
    """
    schemas = {}
    expected = None
    for i, item in enumerate(records):
        label = str(item.get("path", f"batch[{i}]"))
        key = ("path", label) if "path" in item else ("object", id(item))
        if key not in schemas:
            record = item
            if "path" in item:
                container = load_file(item["path"])
                if len(container) != 1:
                    raise ValueError(f"{label}: directory entries must contain exactly one motion")
                record = next(iter(container.values()))
            schemas[key] = motion_schema(record, label)
        schema = schemas[key]
        if expected is None:
            expected = schema
        if schema != expected:
            raise ValueError(f"{label}: mixed SoftSONIC schema {schema}, expected {expected}; "
                             "construct zero-wrench entries with q_aug=q_ref and zero metadata")
    if expected is None:
        raise ValueError("Cannot load an empty motion batch")
    # This SONIC transform runs even during evaluation, but only shifts q_ref.
    if any(expected) and cfg.get("zero_root_xy", False):
        raise ValueError("SoftSONIC zero_root_xy would shift q_ref without q_aug/field")
    if any(expected) and not is_evaluation:
        incompatible = [name for name in ("freeze_frame_aug", "randomize_heading",
                        "randomize_wrist_poses", "cat_upper_body_poses",
                        "randomize_upper_body_poses") if cfg.get(name, False)]
        if incompatible:
            raise ValueError("SoftSONIC q_ref/q_aug/field transformations are not synchronized: "
                             + ", ".join(incompatible))
    return expected


def get_worker_result(queue, workers):
    """Propagate dead FK workers instead of waiting forever for a missing result."""
    while True:
        try:
            return queue.get(timeout=1.0)
        except Empty:
            failed = [(w.pid, w.exitcode) for w in workers if w.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Motion FK workers failed (pid, exitcode): {failed}")
            if workers and all(w.exitcode is not None for w in workers):
                raise RuntimeError("Motion FK workers exited without returning all results")
