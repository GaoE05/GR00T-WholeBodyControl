"""Pure-torch indexing and sampling for SoftSONIC reset policy C.

The runtime motion library supplies two masks on its already-resampled frame
grid: whether either force-field stiffness is active, and whether q_aug has
recovered to q_ref in both pose and velocity.  This module deliberately knows
nothing about source-frame rates or Isaac Sim, which keeps the legality rules
testable on CPU and prevents 30 Hz frame counts from leaking into a 50 Hz
command cursor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreForceResetIndex:
    """CSR-style legal-start index grouped by force event."""

    event_motion_ids: torch.Tensor
    event_starts: torch.Tensor
    event_ends: torch.Tensor
    candidate_offsets: torch.Tensor
    candidate_steps: torch.Tensor
    total_events: int
    rejected_tail_events: int
    rejected_no_candidate_events: int
    fps: float
    window_steps: int
    min_lead_steps: int
    min_tail_steps: int

    @property
    def num_events(self) -> int:
        return int(self.event_motion_ids.numel())

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_steps.numel())

    def sample(
        self,
        num_samples: int,
        *,
        motion_ids: torch.Tensor | None = None,
        event_weights: torch.Tensor | None = None,
        lead_sampling: str = "uniform",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an event first, then a legal lead uniformly within it.

        ``motion_ids`` conditions each output row on an already-selected local
        motion (paired/evaluation and multi-object entry points).  Without it,
        events are uniform unless adaptive ``event_weights`` are supplied.
        Candidate multiplicity never changes an event's probability.
        """
        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}")
        if lead_sampling not in {"uniform", "earliest"}:
            raise ValueError(
                "lead_sampling must be 'uniform' or 'earliest', "
                f"got {lead_sampling!r}"
            )
        device = self.event_motion_ids.device
        if num_samples == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty
        if self.num_events == 0:
            raise RuntimeError("reset C has no legal pre-force events in the loaded motion batch")

        weights = None
        if event_weights is not None:
            if event_weights.shape != (self.num_events,):
                raise ValueError(
                    f"event_weights must have shape ({self.num_events},), "
                    f"got {tuple(event_weights.shape)}"
                )
            weights = event_weights.to(device=device, dtype=torch.float64)
            if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("event_weights must be finite, non-negative, and have positive sum")

        if motion_ids is None:
            if weights is None:
                chosen_events = torch.randint(self.num_events, (num_samples,), device=device)
            else:
                chosen_events = torch.multinomial(weights, num_samples, replacement=True)
        else:
            requested = motion_ids.to(device=device, dtype=torch.long)
            if requested.shape != (num_samples,):
                raise ValueError(
                    f"motion_ids must have shape ({num_samples},), got {tuple(requested.shape)}"
                )
            chosen_events = torch.empty(num_samples, dtype=torch.long, device=device)
            for motion_id in requested.unique():
                rows = torch.nonzero(requested == motion_id, as_tuple=False).flatten()
                eligible = torch.nonzero(
                    self.event_motion_ids == motion_id, as_tuple=False
                ).flatten()
                if eligible.numel() == 0:
                    raise RuntimeError(
                        f"reset C motion {int(motion_id)} has no legal pre-force event"
                    )
                if weights is None:
                    picks = torch.randint(eligible.numel(), (rows.numel(),), device=device)
                else:
                    local_weights = weights[eligible]
                    if local_weights.sum() <= 0:
                        raise RuntimeError(
                            f"reset C motion {int(motion_id)} has zero adaptive event mass"
                        )
                    picks = torch.multinomial(local_weights, rows.numel(), replacement=True)
                chosen_events[rows] = eligible[picks]

        starts = self.candidate_offsets[chosen_events]
        counts = self.candidate_offsets[chosen_events + 1] - starts
        if lead_sampling == "earliest":
            # Candidate steps are stored in ascending time order, so the first
            # candidate gives the longest recovered/no-field history available
            # inside the configured window.
            candidate_rows = starts
        else:
            candidate_rows = starts + (
                torch.rand(num_samples, device=device) * counts
            ).floor().long()
        sampled_steps = self.candidate_steps[candidate_rows]
        sampled_motion_ids = self.event_motion_ids[chosen_events]
        return sampled_motion_ids, sampled_steps, chosen_events


def _seconds_to_steps(seconds: float, fps: float, name: str) -> int:
    if seconds < 0:
        raise ValueError(f"{name} must be non-negative, got {seconds}")
    exact = seconds * fps
    rounded = int(round(exact))
    if abs(exact - rounded) > 1e-6:
        raise ValueError(f"{name}={seconds}s is not an integer number of frames at {fps} Hz")
    return rounded


def build_pre_force_reset_index(
    active: torch.Tensor,
    recovered: torch.Tensor,
    motion_num_frames: torch.Tensor,
    *,
    fps: float,
    window_s: float,
    min_lead_s: float,
    min_tail_s: float,
) -> PreForceResetIndex:
    """Build legal starts from the contiguous inactive gap before each event.

    A legal candidate is in the immediately preceding inactive gap, within the
    configured time window, at least ``min_lead_s`` before onset, and recovered
    at the candidate frame itself. It need not remain recovered up to onset: the
    current dataset's q_aug velocity leads field activation by 4--9 runtime
    frames. An event must close and leave the requested post-event tail. Both
    masks are expected after all loading and resampling.
    """
    if active.ndim != 1 or recovered.ndim != 1 or active.shape != recovered.shape:
        raise ValueError("active and recovered must be same-length 1-D tensors")
    if motion_num_frames.ndim != 1:
        raise ValueError("motion_num_frames must be a 1-D tensor")
    if int(motion_num_frames.sum()) != active.numel():
        raise ValueError(
            f"frame lengths sum to {int(motion_num_frames.sum())}, but masks have {active.numel()}"
        )
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    window_steps = _seconds_to_steps(window_s, fps, "window_s")
    min_lead_steps = _seconds_to_steps(min_lead_s, fps, "min_lead_s")
    min_tail_steps = _seconds_to_steps(min_tail_s, fps, "min_tail_s")
    if window_steps < min_lead_steps:
        raise ValueError("window_s must be at least min_lead_s")

    active = active.bool()
    recovered = recovered.bool()
    event_motion_ids: list[int] = []
    event_starts: list[int] = []
    event_ends: list[int] = []
    candidate_chunks: list[torch.Tensor] = []
    candidate_offsets = [0]
    total_events = 0
    rejected_tail_events = 0
    rejected_no_candidate_events = 0
    frame_offset = 0

    for motion_id, num_frames_tensor in enumerate(motion_num_frames):
        num_frames = int(num_frames_tensor)
        motion_active = active[frame_offset : frame_offset + num_frames]
        motion_recovered = recovered[frame_offset : frame_offset + num_frames]
        padded = torch.cat(
            [
                torch.zeros(1, dtype=torch.bool, device=active.device),
                motion_active,
                torch.zeros(1, dtype=torch.bool, device=active.device),
            ]
        )
        starts = torch.nonzero(~padded[:-1] & padded[1:], as_tuple=False).flatten()
        ends = torch.nonzero(padded[:-1] & ~padded[1:], as_tuple=False).flatten()
        if starts.numel() != ends.numel():
            raise RuntimeError(f"force-event edge mismatch in motion {motion_id}")

        previous_event_end = 0
        for start_tensor, end_tensor in zip(starts, ends, strict=True):
            start, end = int(start_tensor), int(end_tensor)
            total_events += 1
            low = max(previous_event_end, start - window_steps)
            high = start - min_lead_steps
            previous_event_end = end

            # An event ending at the clip boundary is not a complete event.
            if num_frames - end < min_tail_steps:
                rejected_tail_events += 1
                continue
            if high < low:
                rejected_no_candidate_events += 1
                continue

            candidates = torch.arange(low, high + 1, dtype=torch.long, device=active.device)
            # Approved relaxed definition: the reset frame itself must match
            # q_ref in pose and velocity. Later q_aug motion before field onset
            # is retained rather than silently phase-shifting the source data.
            candidates = candidates[motion_recovered[candidates]]
            if candidates.numel() == 0:
                rejected_no_candidate_events += 1
                continue

            event_motion_ids.append(motion_id)
            event_starts.append(start)
            event_ends.append(end)
            candidate_chunks.append(candidates)
            candidate_offsets.append(candidate_offsets[-1] + candidates.numel())
        frame_offset += num_frames

    device = active.device
    return PreForceResetIndex(
        event_motion_ids=torch.tensor(event_motion_ids, dtype=torch.long, device=device),
        event_starts=torch.tensor(event_starts, dtype=torch.long, device=device),
        event_ends=torch.tensor(event_ends, dtype=torch.long, device=device),
        candidate_offsets=torch.tensor(candidate_offsets, dtype=torch.long, device=device),
        candidate_steps=(
            torch.cat(candidate_chunks)
            if candidate_chunks
            else torch.empty(0, dtype=torch.long, device=device)
        ),
        total_events=total_events,
        rejected_tail_events=rejected_tail_events,
        rejected_no_candidate_events=rejected_no_candidate_events,
        fps=float(fps),
        window_steps=window_steps,
        min_lead_steps=min_lead_steps,
        min_tail_steps=min_tail_steps,
    )


def project_adaptive_bins_to_events(
    index: PreForceResetIndex,
    bin_motion_ids: torch.Tensor,
    bin_starts: torch.Tensor,
    bin_ends: torch.Tensor,
    bin_weights: torch.Tensor,
) -> torch.Tensor:
    """Project adaptive bin mass onto event-conditioned rollout spans.

    A span runs from an event's earliest legal candidate through field end.
    Each bin's mass is split equally between overlapping spans. Bins without a
    reachable event are discarded and the result is normalized. Thus adaptive
    failure statistics affect event choice, while legal-lead count never does.
    """
    tensors = (bin_motion_ids, bin_starts, bin_ends, bin_weights)
    if any(t.ndim != 1 for t in tensors) or len({t.numel() for t in tensors}) != 1:
        raise ValueError("adaptive bin tensors must be same-length 1-D tensors")
    weights = torch.zeros(
        index.num_events, dtype=torch.float64, device=index.event_motion_ids.device
    )
    event_support_starts = index.candidate_steps[index.candidate_offsets[:-1]]
    # Every indexed event has at least one candidate by construction.
    assert event_support_starts.shape == index.event_starts.shape
    for motion_id, start, end, mass in zip(*tensors, strict=True):
        overlaps = torch.nonzero(
            (index.event_motion_ids == motion_id)
            & (event_support_starts < end)
            & (index.event_ends > start),
            as_tuple=False,
        ).flatten()
        if overlaps.numel() > 0:
            weights[overlaps] += mass.to(weights.dtype) / overlaps.numel()
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise RuntimeError("adaptive bins assign no valid probability mass to reset-C events")
    return weights / weights.sum()
