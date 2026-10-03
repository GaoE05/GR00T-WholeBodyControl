"""Default-off quasi-static diagnostic prototype; deliberately no training reward hook.
Kcmd is desired robot stiffness (N/m), never environmental forcefield stiffness.
Caller must supply verified pre-reset world-frame inputs and certified masks.
"""
import torch


def spring_consistency_diagnostic(force_w, displacement_w, *, enabled=False,
                                  actual_active=None, static_mask=None, hold_mask=None,
                                  feasible_mask=None, force_tick=None, displacement_tick=None,
                                  command_stiffness=45.0, force_scale_N=20.0):
    if force_w.shape != displacement_w.shape or force_w.shape[-1] != 3:
        raise ValueError("force/displacement require matching (...,3) world-frame shape")
    if command_stiffness <= 0 or force_scale_N <= 0:
        raise ValueError("positive robot Kcmd and fixed Newton scale required")
    shape = force_w.shape[:-1]
    valid = torch.zeros(shape, device=force_w.device, dtype=torch.bool)
    zero = torch.zeros(shape, device=force_w.device, dtype=force_w.dtype)
    if not enabled:
        return {"valid": valid, "valid_count": valid.sum(), "squared_error": zero, "mean_squared_error": zero.sum()}
    inputs = (actual_active, static_mask, hold_mask, feasible_mask, force_tick, displacement_tick)
    if any(x is None for x in inputs):
        return {"valid": valid, "valid_count": valid.sum(), "squared_error": zero, "mean_squared_error": zero.sum()}
    if any(x.shape != shape for x in inputs):
        raise ValueError("all masks and tick IDs must match batch shape")
    valid = (actual_active.bool() & static_mask.bool() & hold_mask.bool() & feasible_mask.bool()
             & (force_tick == displacement_tick) & torch.isfinite(force_w).all(-1)
             & torch.isfinite(displacement_w).all(-1))
    safe_force = torch.where(valid[..., None], force_w, torch.zeros_like(force_w))
    safe_displacement = torch.where(valid[..., None], displacement_w, torch.zeros_like(displacement_w))
    squared_error = ((safe_force - command_stiffness * safe_displacement) / force_scale_N).square().sum(-1)
    return {"valid": valid, "valid_count": valid.sum(), "squared_error": squared_error,
            "mean_squared_error": squared_error.sum() / valid.sum().clamp_min(1)}
