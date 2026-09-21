"""CPU-only numerical validation of reset C on a real SoftSONIC motion library."""

import argparse
import json
from pathlib import Path

from easydict import EasyDict
from loguru import logger
import torch

from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot


ROOT = Path(__file__).resolve().parents[2]


def motion_lib_config(motion_file: Path) -> EasyDict:
    return EasyDict(
        {
            "motion_file": str(motion_file),
            "asset": {
                "assetRoot": str(
                    ROOT / "gear_sonic/data/assets/robot_description/mjcf"
                ),
                "assetFileName": "g1_29dof_rev_1_0.xml",
                "urdfFileName": "",
            },
            "extend_config": [],
            "target_fps": 50,
            "step_dt": 0.02,
            "multi_thread": False,
            "smpl_motion_file": None,
            "filter_motion_keys": None,
            "adaptive_sampling": {
                "enable": True,
                "bin_size": 50,
                "sequence_length_agnostic": True,
                "init_num_failures": 1,
                "uniform_sampling_rate": 0.1,
                # Reset C replaces this arbitrary-frame shift with its own
                # legal event/lead sampling, but preserves the bin weights.
                "pre_failure_sample_window": 200,
                "use_failure_rate_decay": False,
                "decay_gamma": 0.8,
                "adp_samp_failure_rate_max_over_mean": 50.0,
            },
        }
    )


def validate(motion_file: Path, adaptive_samples: int) -> dict:
    logger.remove()
    lib = MotionLibRobot(
        motion_lib_config(motion_file), num_envs=10, device=torch.device("cpu")
    )
    lib.load_motions_for_training(max_num_seqs=10)
    lib.configure_pre_force_reset(
        window_s=1.0,
        min_lead_s=0.2,
        min_tail_s=0.2,
        pose_tolerance=1.0e-5,
        velocity_tolerance=1.0e-4,
        field_tolerance=0.0,
    )
    initial_index = lib.pre_force_reset_index
    lib.load_motions_for_evaluation(start_idx=0)
    reload_rebuilt_index = lib.pre_force_reset_index is not initial_index
    assert reload_rebuilt_index

    index = lib.pre_force_reset_index
    counts = index.candidate_offsets[1:] - index.candidate_offsets[:-1]
    flat_candidates = index.candidate_steps + torch.repeat_interleave(
        lib.length_starts[index.event_motion_ids], counts
    )
    field = lib._motion_softsonic[flat_candidates, 7:9]  # noqa: SLF001
    pose_error = lib.pre_force_reset_pose_error[flat_candidates]
    velocity_error = lib.pre_force_reset_velocity_error[flat_candidates]
    leads = (
        torch.repeat_interleave(index.event_starts, counts) - index.candidate_steps
    )

    assert index.total_events == index.num_events == 280
    assert index.num_candidates == 8366
    assert index.rejected_tail_events == 0
    assert index.rejected_no_candidate_events == 0
    assert (field == 0).all()
    assert (pose_error <= 1.0e-5).all()
    assert (velocity_error <= 1.0e-4).all()
    assert int(leads.min()) == 10 and int(leads.max()) == 50

    adaptive_weights = lib._adaptive_pre_force_event_weights()  # noqa: SLF001
    assert adaptive_weights.shape == (280,)
    assert torch.isfinite(adaptive_weights).all() and (adaptive_weights > 0).all()

    torch.manual_seed(20260921)
    sampled_ids, sampled_steps = lib.sample_pre_force_reset(
        adaptive_samples, adaptive=True
    )
    sampled_pairs = torch.stack([sampled_ids, sampled_steps], dim=1)
    legal_pairs = torch.stack(
        [torch.repeat_interleave(index.event_motion_ids, counts), index.candidate_steps],
        dim=1,
    )
    encoding_base = int(lib._motion_num_frames.max()) + 1  # noqa: SLF001
    sampled_codes = sampled_pairs[:, 0] * encoding_base + sampled_pairs[:, 1]
    legal_codes = legal_pairs[:, 0] * encoding_base + legal_pairs[:, 1]
    assert torch.isin(sampled_codes, legal_codes).all()

    per_motion_candidates = torch.zeros(10, dtype=torch.long).scatter_add_(
        0, index.event_motion_ids, counts
    )
    return {
        "runtime_frames": int(lib._motion_num_frames.sum()),  # noqa: SLF001
        "runtime_fps": index.fps,
        "events_total": index.total_events,
        "events_covered": index.num_events,
        "coverage": index.num_events / index.total_events,
        "legal_candidates": index.num_candidates,
        "expected_candidates": 8366,
        "candidate_difference": index.num_candidates - 8366,
        "window_steps": index.window_steps,
        "min_lead_steps": index.min_lead_steps,
        "min_tail_steps": index.min_tail_steps,
        "lead_min": int(leads.min()),
        "lead_max": int(leads.max()),
        "candidate_count_min": int(counts.min()),
        "candidate_count_median": float(counts.float().median()),
        "candidate_count_max": int(counts.max()),
        "candidate_field_abs_max": float(field.abs().max()),
        "candidate_pose_error_max": float(pose_error.max()),
        "candidate_velocity_error_max": float(velocity_error.max()),
        "per_motion_events": torch.bincount(
            index.event_motion_ids, minlength=10
        ).tolist(),
        "per_motion_candidates": per_motion_candidates.tolist(),
        "adaptive_positive_events": int((adaptive_weights > 0).sum()),
        "adaptive_weight_sum": float(adaptive_weights.sum()),
        "adaptive_sample_count": adaptive_samples,
        "adaptive_samples_all_legal": True,
        "motion_reload_rebuilt_index": reload_rebuilt_index,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_file", type=Path)
    parser.add_argument("--adaptive-samples", type=int, default=100_000)
    args = parser.parse_args()
    result = validate(args.motion_file, args.adaptive_samples)
    print("RESET_C_VALIDATION " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
