"""CPU regression tests for Actor standard-deviation reads."""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from gear_sonic.trl.modules.actor_critic_modules import Actor


class _Config(dict):
    def __getattr__(self, key):
        return self[key]


class _IdentityBackbone(nn.Module):
    def forward(self, value, **_kwargs):
        return value


def _make_actor(values, *, use_log_std=False, clamp_noise_std=True):
    actor = Actor.__new__(Actor)
    nn.Module.__init__(actor)
    actor.algo_config = _Config(
        use_clampped_std=True,
        std_clamp_min=0.05,
        std_clamp_max=0.8,
    )
    actor.use_log_std = use_log_std
    if use_log_std:
        actor.log_std = nn.Parameter(torch.as_tensor(values, dtype=torch.float32).clone())
    else:
        actor.std = nn.Parameter(torch.as_tensor(values, dtype=torch.float32).clone())
    actor.clamp_noise_std = clamp_noise_std
    actor.max_noise_std = 0.5
    actor.input_key = "actor_obs"
    actor.running_mean_std = None
    actor.use_batch_norm = False
    actor.input_obs_dict = False
    actor.has_aux_loss = False
    actor.actor_module = _IdentityBackbone()
    actor.distribution = None
    actor.eval()
    return actor


def _state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _legacy_projected_std(value):
    projected = value.detach().clone()
    with torch.no_grad():
        projected.clamp_(min=0.05, max=0.8)
        projected.clamp_(max=0.5)
    return projected


def test_distribution_read_is_state_preserving_and_numerically_equivalent():
    raw_std = torch.tensor([-0.1, 0.1, 0.6, 1.2])
    actor = _make_actor(raw_std)
    before = _state_hash(actor)

    actor.update_distribution({"actor_obs": torch.zeros(3, raw_std.numel())})

    assert _state_hash(actor) == before
    expected = _legacy_projected_std(raw_std).expand(3, -1)
    torch.testing.assert_close(actor.action_std, expected, rtol=0, atol=0)
    torch.testing.assert_close(actor.std, raw_std, rtol=0, atol=0)


def test_projected_std_keeps_legacy_identity_gradient_at_boundaries():
    raw_std = torch.tensor([-0.1, 0.1, 0.6, 1.2])
    actor = _make_actor(raw_std)

    actor.get_std.sum().backward()

    # The old getter first projected under no_grad and then differentiated the
    # parameter itself, so all entries had derivative one, including entries
    # outside the configured interval.
    torch.testing.assert_close(actor.std.grad, torch.ones_like(raw_std), rtol=0, atol=0)
    torch.testing.assert_close(actor.std, raw_std, rtol=0, atol=0)


def test_explicit_post_step_projection_matches_legacy_multi_step_trajectory():
    initial = torch.tensor([0.49, 0.06])
    actor = _make_actor(initial)
    legacy_std = nn.Parameter(initial.clone())
    actor_optimizer = torch.optim.Adam([actor.std], lr=0.05)
    legacy_optimizer = torch.optim.Adam([legacy_std], lr=0.05)
    gradient = torch.tensor([-1.0, 1.0])

    # The trainer projects once before its first rollout/forward.
    actor.project_std_()
    for _ in range(6):
        # This is where the old getter performed its hidden write.
        with torch.no_grad():
            legacy_std.clamp_(min=0.05, max=0.8)
            legacy_std.clamp_(max=0.5)

        torch.testing.assert_close(actor.std, legacy_std, rtol=0, atol=0)
        actor_optimizer.zero_grad()
        legacy_optimizer.zero_grad()
        (actor.get_std * gradient).sum().backward()
        (legacy_std * gradient).sum().backward()
        actor_optimizer.step()
        legacy_optimizer.step()
        # The new trainer makes the same projection explicit immediately
        # after the update, before any subsequent rollout or PPO forward.
        actor.project_std_()
        assert torch.all(actor.std >= 0.05)
        assert torch.all(actor.std <= 0.5)

    with torch.no_grad():
        legacy_std.clamp_(min=0.05, max=0.8)
        legacy_std.clamp_(max=0.5)
    torch.testing.assert_close(actor.std, legacy_std, rtol=0, atol=0)
    for key, value in actor_optimizer.state[actor.std].items():
        torch.testing.assert_close(value, legacy_optimizer.state[legacy_std][key], rtol=0, atol=0)


def test_legacy_std_checkpoint_loads_strictly_without_schema_changes():
    source = _make_actor([0.04, 0.2, 0.50002408, 0.9])
    legacy_state = source.state_dict()
    target = _make_actor([0.1, 0.1, 0.1, 0.1])

    incompatible = target.load_state_dict(legacy_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(target.std, source.std, rtol=0, atol=0)
    before = _state_hash(target)
    torch.testing.assert_close(target.get_std, _legacy_projected_std(source.std), rtol=0, atol=0)
    assert _state_hash(target) == before


def test_invalid_log_std_uses_safe_read_without_repairing_checkpoint_state():
    values = torch.tensor([float("nan"), float("inf"), -1.0])
    actor = _make_actor(values, use_log_std=True, clamp_noise_std=False)
    before = _state_hash(actor)

    std = actor.get_std

    torch.testing.assert_close(std[:2], torch.full((2,), 0.5), rtol=0, atol=0)
    torch.testing.assert_close(std[2], torch.exp(values[2]), rtol=0, atol=0)
    assert _state_hash(actor) == before

    actor.project_std_()
    torch.testing.assert_close(actor.log_std, torch.full((3,), torch.log(torch.tensor(0.5))))
