"""CPU tests for event-indexed SoftSONIC reset policy C."""

from pathlib import Path

import torch

from gear_sonic.utils.motion_lib.pre_force_reset import (
    build_pre_force_reset_index,
    project_adaptive_bins_to_events,
)


def make_index(active, recovered=None, *, fps=50, window_s=1.0, lead_s=0.2, tail_s=0.2):
    active = torch.as_tensor(active, dtype=torch.bool)
    if recovered is None:
        recovered = torch.ones_like(active)
    return build_pre_force_reset_index(
        active,
        torch.as_tensor(recovered, dtype=torch.bool),
        torch.tensor([len(active)]),
        fps=fps,
        window_s=window_s,
        min_lead_s=lead_s,
        min_tail_s=tail_s,
    )


def test_seconds_map_to_runtime_grid():
    active_50 = torch.zeros(120, dtype=torch.bool)
    active_50[70:90] = True
    index_50 = make_index(active_50)
    assert index_50.window_steps == 50
    assert index_50.min_lead_steps == 10
    assert index_50.candidate_steps.tolist() == list(range(20, 61))

    active_30 = torch.zeros(80, dtype=torch.bool)
    active_30[45:60] = True
    index_30 = make_index(active_30, fps=30)
    assert index_30.window_steps == 30
    assert index_30.min_lead_steps == 6
    assert index_30.candidate_steps.tolist() == list(range(15, 40))


def test_contiguous_idle_gap_and_candidate_recovery():
    active = torch.zeros(100, dtype=torch.bool)
    active[10:20] = True
    active[70:80] = True
    recovered = torch.ones(100, dtype=torch.bool)
    recovered[53:56] = False
    index = make_index(active, recovered)
    # First event has only the frame exactly 0.2 s before onset. The second
    # cannot cross the previous event; only the unrecovered frames themselves
    # are rejected under the approved relaxed definition.
    assert index.event_starts.tolist() == [10, 70]
    expected = [0, *range(20, 53), *range(56, 61)]
    assert index.candidate_steps.tolist() == expected
    assert recovered[index.candidate_steps].all()


def test_tail_and_missing_candidates_are_explicit():
    active = torch.zeros(100, dtype=torch.bool)
    active[2:10] = True  # no 0.2 s lead
    active[90:] = True  # does not close / has no tail
    index = make_index(active)
    assert index.total_events == 2
    assert index.num_events == 0
    assert index.rejected_no_candidate_events == 1
    assert index.rejected_tail_events == 1


def test_event_first_sampling_does_not_weight_longer_gaps():
    active = torch.zeros(180, dtype=torch.bool)
    active[10:20] = True
    active[120:130] = True
    index = make_index(active)
    counts = index.candidate_offsets[1:] - index.candidate_offsets[:-1]
    assert counts.tolist() == [1, 41]
    torch.manual_seed(7)
    _, steps, events = index.sample(20000)
    fraction_second = (events == 1).float().mean().item()
    assert 0.48 < fraction_second < 0.52
    assert ((steps[events == 0]) == 0).all()


def test_earliest_sampling_uses_first_legal_candidate_per_event():
    active = torch.zeros(180, dtype=torch.bool)
    active[10:20] = True
    active[120:130] = True
    index = make_index(active)
    requested = torch.tensor([0, 0])
    torch.manual_seed(11)
    _, steps, events = index.sample(
        2, motion_ids=requested, lead_sampling="earliest"
    )
    expected = index.candidate_steps[index.candidate_offsets[events]]
    torch.testing.assert_close(steps, expected)

    try:
        index.sample(1, lead_sampling="latest")
    except ValueError as exc:
        assert "uniform" in str(exc) and "earliest" in str(exc)
    else:
        raise AssertionError("invalid lead_sampling must be rejected")


def test_conditioned_sampling_and_adaptive_projection():
    active = torch.zeros(240, dtype=torch.bool)
    active[20:30] = True
    active[100:110] = True
    active[180:190] = True
    index = build_pre_force_reset_index(
        active,
        torch.ones_like(active),
        torch.tensor([120, 120]),
        fps=50,
        window_s=1.0,
        min_lead_s=0.2,
        min_tail_s=0.2,
    )
    requested = torch.tensor([0, 1, 1, 0])
    motion_ids, _, _ = index.sample(4, motion_ids=requested)
    torch.testing.assert_close(motion_ids, requested)

    # These bins overlap only legal pre-event rollout spans, not the active
    # fields themselves. Adaptive mass still reaches the corresponding event:
    # this is direct event reweighting, not frame sampling + snapping.
    weights = project_adaptive_bins_to_events(
        index,
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 50, 10]),
        torch.tensor([15, 95, 55]),
        torch.tensor([0.2, 0.3, 0.5]),
    )
    torch.testing.assert_close(weights, torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64))


def test_all_command_sampling_entry_points_use_reset_c_sampler():
    source = (
        Path(__file__).resolve().parents[1]
        / "envs/manager_env/mdp/commands.py"
    ).read_text()
    # configure + initial assignment + forward_motion_samples + evaluating +
    # paired + multi-object + ordinary reset.
    assert source.count("sample_pre_force_reset(") == 6
    assert "configure_pre_force_reset(" in source
    assert "self.motion_ids,\n                    self.motion_start_time_steps" in source
    assert "sampled_times = torch.zeros(" in source
    assert "new_motion_id = self.motion_lib.sample_motions(1)[0]" in source
    motion_lib_source = (
        Path(__file__).resolve().parents[1]
        / "utils/motion_lib/motion_lib_base.py"
    ).read_text()
    assert "field_stiffness = self._motion_softsonic[:, 7:9]" in motion_lib_source
    assert 'hasattr(self, "_pre_force_reset_config")' in motion_lib_source
    assert "Pure-zero motions retain both values exactly" in motion_lib_source
    assert 'lead_sampling: str = "earliest"' in source



if __name__ == "__main__":
    for name, value in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("RESET-C CPU tests PASS")
