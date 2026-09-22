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
        motion_lib_config(motion_file), num_envs=1024, device=torch.device("cpu")
    )
    lib.load_motions_for_training(max_num_seqs=lib._num_unique_motions)  # noqa: SLF001
    lib.configure_pre_force_reset(
        window_s=1.0,
        min_lead_s=0.2,
        min_tail_s=0.2,
        pose_tolerance=1.0e-5,
        velocity_tolerance=1.0e-4,
        field_tolerance=0.0,
        lead_sampling="earliest",
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

    num_motions = int(lib._motion_num_frames.numel())  # noqa: SLF001
    num_zero_motions = int((~lib.pre_force_motion_has_events).sum())
    expected = {
        (10, 0): (280, 8366),
        (16, 8): (222, 6631),
    }.get((num_motions, num_zero_motions))
    assert expected is not None, (num_motions, num_zero_motions)
    expected_events, expected_candidates = expected

    assert index.total_events == index.num_events == expected_events
    assert index.num_candidates == expected_candidates
    assert index.rejected_tail_events == 0
    assert index.rejected_no_candidate_events == 0
    assert (field == 0).all()
    assert (pose_error <= 1.0e-5).all()
    assert (velocity_error <= 1.0e-4).all()
    assert int(leads.min()) == 10 and int(leads.max()) == 50

    adaptive_weights = lib._adaptive_pre_force_event_weights()  # noqa: SLF001
    assert adaptive_weights.shape == (expected_events,)
    assert torch.isfinite(adaptive_weights).all() and (adaptive_weights > 0).all()

    # Actual C sampling starts with the exact A motion/time draw.  C must keep
    # pure-zero rows byte-for-byte while replacing only force timestamps.
    torch.manual_seed(20260921)
    base_ids = lib.sample_motions(adaptive_samples)
    base_steps = lib.sample_time_steps(base_ids, truncate_time=None)
    sampled_ids, sampled_steps = lib.sample_pre_force_reset(
        base_ids, base_steps, adaptive=False
    )
    torch.testing.assert_close(sampled_ids, base_ids)
    sampled_zero = ~lib.pre_force_motion_has_events[sampled_ids]
    torch.testing.assert_close(sampled_steps[sampled_zero], base_steps[sampled_zero])

    earliest_steps = index.candidate_steps[index.candidate_offsets[:-1]]
    earliest_pairs = torch.stack([index.event_motion_ids, earliest_steps], dim=1)
    encoding_base = int(lib._motion_num_frames.max()) + 1  # noqa: SLF001
    sampled_force_codes = (
        sampled_ids[~sampled_zero] * encoding_base + sampled_steps[~sampled_zero]
    )
    earliest_codes = earliest_pairs[:, 0] * encoding_base + earliest_pairs[:, 1]
    assert torch.isin(sampled_force_codes, earliest_codes).all()
    sampled_event_coverage = int(torch.unique(sampled_force_codes).numel())
    assert sampled_event_coverage == expected_events

    expected_zero_fraction = float(
        lib._sampling_batch_prob[~lib.pre_force_motion_has_events].sum()  # noqa: SLF001
    )
    sampled_zero_fraction = float(sampled_zero.float().mean())
    assert abs(sampled_zero_fraction - expected_zero_fraction) < 0.01

    torch.manual_seed(20260922)
    adaptive_ids, adaptive_base_steps = lib.sample_motion_ids_and_time_steps(10_000)
    adaptive_c_ids, adaptive_c_steps = lib.sample_pre_force_reset(
        adaptive_ids, adaptive_base_steps, adaptive=True
    )
    torch.testing.assert_close(adaptive_c_ids, adaptive_ids)
    adaptive_zero = ~lib.pre_force_motion_has_events[adaptive_ids]
    torch.testing.assert_close(
        adaptive_c_steps[adaptive_zero], adaptive_base_steps[adaptive_zero]
    )
    adaptive_force_codes = (
        adaptive_c_ids[~adaptive_zero] * encoding_base
        + adaptive_c_steps[~adaptive_zero]
    )
    assert torch.isin(adaptive_force_codes, earliest_codes).all()

    # Quantify the stronger history notion: time until q_aug first ceases to
    # match q_ref.  The approved earliest policy keeps all events and must
    # reproduce the 6/222 (2.70%) result on the mixed long-training dataset.
    strong_stable_steps = []
    for motion_id, candidate, event_start in zip(
        index.event_motion_ids.tolist(), earliest_steps.tolist(), index.event_starts.tolist()
    ):
        offset = int(lib.length_starts[motion_id])
        recovered_prefix = lib.pre_force_reset_recovered[
            offset + candidate : offset + event_start
        ]
        lost = torch.nonzero(~recovered_prefix, as_tuple=False).flatten()
        strong_stable_steps.append(int(lost[0]) if lost.numel() else len(recovered_prefix))
    strong_stable_steps = torch.tensor(strong_stable_steps)
    strong_stability_short = int((strong_stable_steps < index.min_lead_steps).sum())
    if (num_motions, num_zero_motions) == (16, 8):
        assert strong_stability_short == 6

    per_motion_candidates = torch.zeros(num_motions, dtype=torch.long).scatter_add_(
        0, index.event_motion_ids, counts
    )
    return {
        "runtime_frames": int(lib._motion_num_frames.sum()),  # noqa: SLF001
        "runtime_fps": index.fps,
        "events_total": index.total_events,
        "events_covered": index.num_events,
        "coverage": index.num_events / index.total_events,
        "legal_candidates": index.num_candidates,
        "expected_candidates": expected_candidates,
        "candidate_difference": index.num_candidates - expected_candidates,
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
            index.event_motion_ids, minlength=num_motions
        ).tolist(),
        "per_motion_candidates": per_motion_candidates.tolist(),
        "adaptive_positive_events": int((adaptive_weights > 0).sum()),
        "adaptive_weight_sum": float(adaptive_weights.sum()),
        "adaptive_sample_count": 10_000,
        "adaptive_samples_all_legal": True,
        "adaptive_motion_ids_preserved": True,
        "adaptive_zero_fallback_steps_preserved": True,
        "lead_sampling": lib.pre_force_reset_lead_sampling,
        "sampled_event_coverage": sampled_event_coverage,
        "num_motions": num_motions,
        "num_force_motions": int(lib.pre_force_motion_has_events.sum()),
        "num_zero_motions": num_zero_motions,
        "expected_zero_fraction": expected_zero_fraction,
        "sampled_zero_fraction": sampled_zero_fraction,
        "zero_sample_count": int(sampled_zero.sum()),
        "zero_fallback_steps_preserved": True,
        "motion_ids_preserved": True,
        "strong_stability_short_count": strong_stability_short,
        "strong_stability_short_fraction": strong_stability_short / expected_events,
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
