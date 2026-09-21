"""Online-first W&B initialization with an explicit offline fallback."""
from __future__ import annotations

import os


PROXY_KEYS = (
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY",
)


def init_wandb_online_first(wandb_module, **init_kwargs):
    """Initialize an online run, falling back to offline only after failure.

    Returns ``(run, status)``.  The status is suitable for the experiment meta
    file, so an offline fallback can never be mistaken for online tracking.
    """
    removed_environment = []
    for key in (*PROXY_KEYS, "WANDB_MODE", "WANDB_DISABLED"):
        if key in os.environ:
            removed_environment.append(key)
            os.environ.pop(key, None)

    fallback_reason = None
    try:
        run = wandb_module.init(
            mode="online",
            settings=wandb_module.Settings(init_timeout=30),
            **init_kwargs,
        )
        if run is None:
            raise RuntimeError("wandb.init(mode=online) returned None")
        actual_mode = "online"
    except Exception as error:  # Transport failures use multiple W&B exception types.
        fallback_reason = f"{type(error).__name__}: {error}"
        print(
            f"WANDB ONLINE FAILED; FALLING BACK TO OFFLINE: {fallback_reason}",
            flush=True,
        )
        try:
            wandb_module.finish(exit_code=1, quiet=True)
        except Exception:
            pass
        run = wandb_module.init(mode="offline", **init_kwargs)
        if run is None:
            raise RuntimeError("wandb.init(mode=offline) returned None") from error
        actual_mode = "offline"

    return run, {
        "required": True,
        "requested_mode": "online",
        "actual_mode": actual_mode,
        "fallback_reason": fallback_reason,
        "removed_environment": removed_environment,
    }
