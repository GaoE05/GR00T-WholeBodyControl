"""CPU-only contract tests for the opt-in pure-zero residual reward."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic/envs/manager_env/mdp/rewards.py"
)
TREE = ast.parse(SOURCE.read_text())
FUNCTIONS = ast.Module(
    body=[
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("latent_residual_l2", "latent_residual_l2_pure_zero")
    ],
    type_ignores=[],
)
NAMESPACE = {"torch": torch, "Path": Path, "json": json, "ManagerBasedRLEnv": object}
exec(compile(FUNCTIONS, str(SOURCE), "exec"), NAMESPACE)
regularizer = NAMESPACE["latent_residual_l2_pure_zero"]


def test_pure_zero_mask_tracks_loaded_keys_not_force_activity(tmp_path):
    path = tmp_path / "data_labels.json"
    path.write_text(json.dumps({"force_a": "force", "zero_a": "zero", "zero_b": "zero"}))
    command = SimpleNamespace(
        motion_lib=SimpleNamespace(curr_motion_keys=["force_a", "zero_a"]),
        motion_ids=torch.tensor([0, 1, 0, 1]),
    )
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        _softsonic_residual=torch.nn.functional.pad(torch.tensor(
            [[0.2, -0.1], [0.3, 0.4], [0.4, 0.5], [0.1, 0.2]]
        ), (0, 62)),
    )
    value = regularizer(env, str(path))
    torch.testing.assert_close(value, torch.tensor([0.0, 0.25, 0.0, 0.05]))
    # A force file stays exempt even if its current force field is inactive.
    command.motion_lib.curr_motion_keys = ["zero_b", "force_a"]
    command.motion_ids = torch.tensor([0, 1, 0, 1])
    value = regularizer(env, str(path))
    torch.testing.assert_close(value, torch.tensor([0.05, 0.0, 0.41, 0.0]))


def test_missing_label_fails_closed(tmp_path):
    path = tmp_path / "data_labels.json"
    path.write_text(json.dumps({"force_a": "force"}))
    command = SimpleNamespace(
        motion_lib=SimpleNamespace(curr_motion_keys=["force_a", "unlabelled"]),
        motion_ids=torch.tensor([0]),
    )
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        _softsonic_residual=torch.zeros(1, 64),
    )
    try:
        regularizer(env, str(path))
    except ValueError as exc:
        assert "unlabelled" in str(exc)
    else:
        raise AssertionError("Missing label was accepted")


def test_zero_weight_leaves_cpu_step_reward_observation_action_bitwise_equal(tmp_path):
    path = tmp_path / "data_labels.json"
    path.write_text(json.dumps({"zero_a": "zero"}))
    command = SimpleNamespace(
        motion_lib=SimpleNamespace(curr_motion_keys=["zero_a"]),
        motion_ids=torch.tensor([0, 0]),
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        _softsonic_residual=torch.zeros(2, 64),
    )
    torch.manual_seed(19)
    observation = torch.randn(2, 8)
    weight = torch.randn(8, 64)
    for _ in range(12):
        action_old = observation @ weight
        action_new = observation @ weight
        env._softsonic_residual = torch.clamp(0.15 * action_new, -0.5, 0.5)
        base_reward = -(action_old.square().sum(-1))
        new_reward = base_reward + 0.0 * regularizer(env, str(path))
        assert torch.equal(action_old, action_new)
        assert torch.equal(base_reward, new_reward)
        next_old = observation + action_old[:, :8] * 0.001
        next_new = observation + action_new[:, :8] * 0.001
        assert torch.equal(next_old, next_new)
        observation = next_old
