"""Reward functions for the manager-based RL environment MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)
import torch

from gear_sonic.envs.manager_env.mdp.commands import (
    ForceTrackingCommand,
    TrackingCommand,
    _get_body_indexes,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    tracking_anchor_pos = None
    tracking_anchor_ori = None
    tracking_relative_body_pos = None
    tracking_relative_body_ori = None
    tracking_relative_body_ori_weighted = None
    tracking_body_linvel = None
    tracking_body_angvel = None
    action_rate_l2 = None
    joint_limit = None
    undesired_contacts = None
    undesired_contacts_no_hands = None
    undesired_contacts_no_ankle_hand = None
    tracking_body_pos = None
    tracking_body_ori = None
    tracking_vr_3point_global = None
    tracking_vr_3point_local = None
    tracking_vr_3point_force = None
    tracking_vr_2wrists_ori_tight = None
    tracking_vr_2wrists_local_ori = None
    tracking_head_local_ori = None
    anti_shake_ang_vel = None
    tracking_vr_5point_local = None
    motion_5point_local_pos = None
    feet_acc = None
    energy_consumption = None
    is_terminated = None
    upright_penalty = None

    # ── SoftSONIC ───────────────────────────────────────────────────────────
    # 对 q_aug 的跟踪项复用上面已有的项名（tracking_* 系列），只在配置里把 func
    # 指向 tracking_compliant_* 版本，因此不需要新字段。下面三项是真正新增的。
    applied_force_tracking = None
    # SoftSONIC：受力连杆位置/朝向跟踪（锚定坐标系），对照 SoftMimic 力控实验
    # 里权重最高的 force_link_keypoint_tracking_local 及其朝向版本
    compliant_force_link_pos = None
    compliant_force_link_ori = None
    applied_torque_tracking = None
    alive = None


def tracking_anchor_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor position tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) position to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel. Smaller values produce
            sharper falloff and stricter tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    diff = command.anchor_pos_w - command.robot_anchor_pos_w
    sq_dist = (diff * diff).sum(dim=-1)
    return torch.exp(-sq_dist / (std * std))


def tracking_anchor_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor orientation tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) orientation to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    angular_err = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return torch.exp(-angular_err.square() / (std * std))


def upright_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_name: str | None = None,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize tilt of bodies away from upright.

    Compute the squared magnitude of the x/y components of the gravity vector
    in each body's local frame, summed across all specified bodies. When a body
    is perfectly upright the local gravity is [0, 0, -1] and the penalty is 0.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        body_name: Single body name (for backwards compatibility).
        body_names: List of body names. If both are None, defaults to ["pelvis"].

    Returns:
        Penalty tensor of shape (num_envs,). Zero when upright, positive when tilted.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]

    if body_names is None:
        body_names = [body_name] if body_name else ["pelvis"]

    total_penalty = torch.zeros(env.num_envs, device=env.device)
    for name in body_names:
        body_idx = robot.body_names.index(name)
        body_quat = robot.data.body_quat_w[:, body_idx]
        g_local = quat_apply(quat_inv(body_quat), command.down_dir)
        total_penalty += g_local[:, 0] ** 2 + g_local[:, 1] ** 2

    return total_penalty


def tracking_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body positions to match the reference motion. The reward is
    the mean squared distance across all tracked bodies, passed through an exponential.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_vr_3point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 3-point tracking reward in world frame using a Gaussian kernel.

    Encourages the robot's 3 VR tracking points (typically left wrist, right wrist,
    head) to match their reference positions in world frame.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_diff = command.robot_vr_3point_pos_w - command.vr_3point_body_pos_w
    per_point_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_point_err.mean(dim=-1) / (std * std))


def tracking_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in world frame.

    Measure the orientation error of 2 wrist bodies against the reference motion,
    similar to tracking_relative_body_ori_error but restricted to wrist links.

    NOTE: The rigid extension defined in vr_3point_body_offset can be skipped for
    orientation error since it does not affect rotations.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_local_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in the anchor's local frame.

    Transform both reference and robot wrist orientations into the anchor (root)
    frame before computing the angular error. This makes the reward invariant to
    global root orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    body_indexes = _get_body_indexes(command, body_names)
    num_bodies = len(body_indexes)

    # reference motion
    ref_wrist_quat_w = command.body_quat_w[:, body_indexes]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, num_bodies, 1)
    ref_wrist_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_wrist_quat_w)

    # robot
    robot_wrist_quat_w = command.robot_body_quat_w[:, body_indexes]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, num_bodies, 1
    )
    robot_wrist_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_wrist_quat_w)

    error = quat_error_magnitude(ref_wrist_quat_local, robot_wrist_quat_local) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def energy_consumption(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize instantaneous mechanical power across the robot joints."""
    if isinstance(asset_cfg, dict):
        robot_name = asset_cfg.get("name", "robot")
    else:
        robot_name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[robot_name]
    return torch.abs(robot.data.applied_torque * robot.data.joint_vel).sum(dim=-1)


def tracking_local_head_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute head orientation tracking reward in the anchor's local frame.

    Transform the head (torso_link) orientation into the anchor's local frame for
    both the reference motion and the robot, then compute the angular error. This
    encourages the robot to match the head-to-root relative orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get the head body index (torso_link)
    head_body_names = ["torso_link"]
    body_indexes = _get_body_indexes(command, head_body_names)

    # reference motion: head orientation in world frame, transformed to anchor's local frame
    ref_head_quat_w = command.body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4)
    ref_head_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_head_quat_w)

    # robot: head orientation in world frame, transformed to anchor's local frame
    robot_head_quat_w = command.robot_body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4)
    robot_head_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_head_quat_w)

    error = quat_error_magnitude(ref_head_quat_local, robot_head_quat_local) ** 2
    return torch.exp(-error.squeeze(-1) / std**2)


def tracking_local_vr_3point_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    point_weights: list[float] | None = None,
):
    """Compute VR 3-point tracking reward in the anchor's local frame.

    Transform tracking points into the anchor (root) local frame before computing
    position error, making the reward invariant to global root position/orientation.
    Supports optional per-point weighting.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        point_weights: Optional weights for each tracking point. Order matches
            vr_3point_body config (typically [left_wrist, right_wrist, head]).
            If None, all points are weighted equally.
            Example: [2, 2, 1] gives wrists 2x importance vs head.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_pos = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    robot_3point_diff = command.robot_vr_3point_pos_w - command.robot_anchor_pos_w[:, None, :]
    robot_3point_pos = quat_apply(quat_inv(robot_root_quat), robot_3point_diff)
    diff = robot_3point_pos - ref_3point_pos
    error = torch.sum(torch.square(diff), dim=-1)  # [num_envs, num_points]

    if point_weights is not None:
        # Weighted mean: sum(w_i * e_i) / sum(w_i)
        weights = torch.tensor(point_weights, dtype=error.dtype, device=error.device)
        weighted_error = (error * weights).sum(dim=-1) / weights.sum()
    else:
        # Simple mean (equal weights)
        weighted_error = error.mean(dim=-1)

    return torch.exp(-weighted_error / std**2)


def tracking_local_vr_5point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 5-point tracking reward in the anchor's local frame.

    Same approach as tracking_local_vr_3point_error but with 5 tracking points
    (e.g., 2 wrists + head + 2 feet).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_5point_diff = command.reward_point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    ref_5point_pos = quat_apply(quat_inv(ref_root_quat), ref_5point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    robot_5point_diff = (
        command.robot_reward_point_body_pos_w - command.robot_anchor_pos_w[:, None, :]
    )
    robot_5point_pos = quat_apply(quat_inv(robot_root_quat), robot_5point_diff)
    diff = robot_5point_pos - ref_5point_pos
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_vr_3point_error_pos_force(
    env: ManagerBasedRLEnv, motion_command_name: str, force_command_name: str, std: float
):
    """Compute VR 3-point tracking reward with force-based compliance correction.

    Add a force-proportional offset to the wrist tracking error so that applied
    external forces shift the tracking target, enabling compliant behavior under
    force perturbations.

    Args:
        env: The environment.
        motion_command_name: Name of the motion tracking command term.
        force_command_name: Name of the force tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    motion_command: TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    diff = motion_command.robot_vr_3point_pos_w - motion_command.vr_3point_body_pos_w
    force_error_wrists = (
        force_command.last_force_applied * force_command.eef_stiffness_buf[:, :, None]
    )
    diff[:, :2] += force_error_wrists
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body orientations to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward using anchor-relative reference positions.

    Use reference body positions that have been shifted to share the robot's anchor
    (root) position, so only the relative pose matters rather than absolute position.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_relative_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_relative_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward using anchor-relative reference orientations.

    Use reference body orientations that have been transformed to share the robot's
    anchor (root) orientation, making the reward invariant to global heading.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_relative_w[:, tracked],
        command.robot_body_quat_w[:, tracked],
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_ori_weighted_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    body_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute anchor-relative body orientation tracking reward with per-body weights.

    Same as tracking_relative_body_ori_error but allows different bodies to contribute
    differently to the mean error. Useful for relaxing tracking on certain joints
    (e.g., wrists during manipulation).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.
        body_weights: Dict mapping body name to weight multiplier. Bodies not listed
            default to 1.0. E.g. {"left_wrist_yaw_link": 0.1} to relax wrist tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    if body_weights is not None:
        tracked_names = [command.cfg.body_names[i] for i in body_indexes]
        weights = torch.tensor(
            [body_weights.get(name, 1.0) for name in tracked_names],
            device=error.device,
            dtype=error.dtype,
        )
        weighted_error = (error * weights).sum(-1) / weights.sum()
    else:
        weighted_error = error.mean(-1)
    return torch.exp(-weighted_error / std**2)


def tracking_body_linvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body linear velocity tracking reward using a Gaussian kernel.

    Encourages tracked body linear velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_lin_vel_w[:, tracked] - command.robot_body_lin_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_body_angvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body angular velocity tracking reward using a Gaussian kernel.

    Encourages tracked body angular velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_ang_vel_w[:, tracked] - command.robot_body_ang_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def anti_shake_ang_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize excessive angular velocity on selected bodies with a deadzone.

    Discourage high-frequency jitter on small links (wrists, head) while allowing
    normal intentional motion within the threshold. Speeds below the threshold
    incur zero penalty.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        threshold: Angular velocity deadzone (rad/s). No penalty below this.
        body_names: Bodies to penalize. If None, uses all tracked bodies.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # [E, B, 3]
    ang_vel = command.robot_body_ang_vel_w[:, body_indexes]
    # magnitude per body: [E, B]
    speed = torch.linalg.norm(ang_vel, dim=-1)
    # deadzone then square: [E, B]
    excess = torch.relu(speed - threshold)
    penalty = (excess * excess).mean(dim=-1)
    return penalty


# ══════════════════════════════════════════════════════════════════════════════
# SoftSONIC：对柔顺目标 q_aug 的跟踪，以及力/力矩跟踪
#
# 设计依据（SoftMimic 论文 III-B 原文）：
#   "the robot observes the original motion target but is rewarded for inferring
#    the force interaction and matching the applicable augmented target"
# 即观测原始参考（真机可得），奖励对齐增强参考（离线 IK 的产物，仅训练时可用）。
#
# q_aug 在无外力时严格等于 q_ref（实测无力帧关节偏差中位 0.000°），所以把跟踪目标
# 整体从 q_ref 换成 q_aug 是安全的，不会在无力时改变行为 —— 也正因如此，不能采用
# "保留 q_ref 奖励再叠加 q_aug 奖励"的做法，那会在有力时制造拉锯，正是 SoftMimic
# 论文指出的 "a purely stiff tracker is a strong local optimum" 陷阱。
#
# 权重取自 SoftMimic Table VI，逐项对照见 notes/reward-mapping.md。
# ══════════════════════════════════════════════════════════════════════════════


def _compliant_relative_ref(command: TrackingCommand):
    """把柔顺目标 q_aug 的连杆位姿重锚到机器人当前 anchor，与 q_ref 侧同一套变换。

    复刻 ``TrackingCommand`` 里 ``body_pos_relative_w`` 的算法，只把 ``body_pos_w``
    换成 ``body_pos_w_aug``。

    **关键**：减去的仍是 ``anchor_pos_w``（q_ref 的 anchor），不是 q_aug 自己的。
    用 q_ref 的 anchor 才能把 ``q_aug - q_ref`` 这个柔顺位移原样保留下来；若用
    q_aug 自己的 anchor，骨盆的柔顺平移会被抵消掉，机器人就不会被奖励去移动骨盆。

    Returns:
        (pos_rel_aug, quat_rel_aug)，形状分别为 (E, B, 3) 和 (E, B, 4)。
    """
    from gear_sonic.trl.utils import torch_transform

    n_bodies = len(command.cfg.body_names)
    anchor_pos_rep = command.anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    anchor_quat_rep = command.anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_pos_rep = command.robot_anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_quat_rep = command.robot_anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)

    delta_pos_w = robot_anchor_pos_rep.clone()
    delta_pos_w[..., 2] = anchor_pos_rep[..., 2]
    delta_ori_w = torch_transform.get_heading_q(
        quat_mul(robot_anchor_quat_rep, quat_inv(anchor_quat_rep))
    )
    pos_rel = delta_pos_w + quat_apply(delta_ori_w, command.body_pos_w_aug - anchor_pos_rep)
    quat_rel = quat_mul(delta_ori_w, command.body_quat_w_aug)
    return pos_rel, quat_rel


def tracking_compliant_relative_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """连杆位置跟踪奖励，目标为柔顺参考 q_aug（对应 q_ref 版的同名奖励）。"""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_rel, _ = _compliant_relative_ref(command)
    pos_diff = pos_rel[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_compliant_relative_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """连杆朝向跟踪奖励，目标为柔顺参考 q_aug。"""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    _, quat_rel = _compliant_relative_ref(command)
    err = quat_error_magnitude(quat_rel[:, tracked], command.robot_body_quat_w[:, tracked])
    return torch.exp(-(err * err).mean(dim=-1) / (std * std))


def tracking_compliant_anchor_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """anchor（骨盆）位置跟踪奖励，目标为柔顺参考。

    骨盆位移是全身协同柔顺的关键体现（实测 q_aug 的骨盆中位移动 10.5cm），
    所以这一项必须瞄 q_aug，否则会与柔顺直接对抗。
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    diff = command.anchor_pos_w_aug - command.robot_anchor_pos_w
    return torch.exp(-(diff * diff).sum(dim=-1) / (std * std))


def tracking_compliant_anchor_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """anchor 朝向跟踪奖励，目标为柔顺参考。"""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    err = quat_error_magnitude(command.anchor_quat_w_aug, command.robot_anchor_quat_w)
    return torch.exp(-(err * err) / (std * std))


def tracking_compliant_local_vr_5point_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """5 点局部跟踪奖励，目标为柔顺参考（对应 q_ref 版权重 2.0 的主项）。

    与 q_ref 版一样把参考与机器人各自变换到自身 anchor 的局部系再比较；参考侧用
    q_ref 的 anchor，使 q_aug 相对 q_ref 的位移得以保留。
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    n_pts = len(command.cfg.reward_point_body)
    ref_diff = command.reward_point_body_pos_w_aug - command.anchor_pos_w[:, None, :]
    ref_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, n_pts, 1)
    ref_local = quat_apply(quat_inv(ref_quat), ref_diff)

    robot_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, n_pts, 1)
    robot_diff = command.robot_reward_point_body_pos_w - command.robot_anchor_pos_w[:, None, :]
    robot_local = quat_apply(quat_inv(robot_quat), robot_diff)

    err = torch.sum(torch.square(robot_local - ref_local), dim=-1)
    return torch.exp(-err.mean(-1) / (std * std))


def _softsonic_wrench_buffers(env: ManagerBasedRLEnv, kind: str):
    """取事件项写入的实际/期望力（或力矩）缓冲。

    首步返回 None：IsaacLab 的管理器顺序是 终止(204) -> 奖励(208) -> 指令推进(232)
    -> 事件(235)，所以第一次算奖励时事件项还没跑过，缓冲尚不存在。这与"压根没启用
    事件项"是两回事，必须区分开 —— 后者是配置错误，静默返回满分会让训练看起来正常
    却完全没有力信号。

    Returns:
        (actual, desired, active) 三元组；首步返回 None。

    Raises:
        RuntimeError: 事件管理器里没有 softsonic_force_field 项（配置错误）。
    """
    actual = getattr(env, f"_softsonic_{kind}_actual", None)
    if actual is not None:
        return (
            actual,
            getattr(env, f"_softsonic_{kind}_desired"),
            env._softsonic_active,  # noqa: SLF001
        )
    terms = getattr(env.event_manager, "active_terms", {})
    names = set()
    for v in (terms.values() if isinstance(terms, dict) else [terms]):
        names.update(v if isinstance(v, (list, tuple)) else [])
    if "softsonic_force_field" not in names:
        raise RuntimeError(
            "力/力矩跟踪奖励要求启用 softsonic_force_field 事件项，但事件管理器里"
            f"没有它。已启用的事件项：{sorted(names)}。"
            "加 '+manager_env/events=tracking/softsonic'。"
        )
    return None


def tracking_compliant_body_linvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """连杆线速度跟踪奖励，目标为柔顺参考 q_aug。

    这一项必须瞄 q_aug，不能留在 q_ref：柔顺退让**本身就是运动**（0.2~1 秒内手腕
    走 24cm），而站立动作的 q_ref 速度接近零。若仍瞄 q_ref，这两个速度项（合计
    权重 2.0）会精确惩罚我们在位姿项与力项上奖励的行为 —— 正是 SoftMimic 指出的
    刚性局部最优拉锯。
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_lin_vel_w_aug[:, tracked] - command.robot_body_lin_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_compliant_body_angvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """连杆角速度跟踪奖励，目标为柔顺参考 q_aug。"""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_ang_vel_w_aug[:, tracked] - command.robot_body_ang_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def _force_link_anchored_target(env, command_name: str, want_quat: bool = False):
    """受力连杆在**锚定坐标系**下的柔顺目标位姿，以及它在本资产里的 body 索引。

    对照 SoftMimic 的 ``force_link_keypoint_tracking_local``
    （``mdp/rewards.py``，力控实验里权重 3.0、σ=0.1，是该实验权重最高的跟踪项）。
    它与我们已有的 5point/body_pos 项有三点不同，不是重复：
      1. 只看**受力连杆**那一个点，而 5point 摊在 5 个点上、信号被稀释；
      2. 在**锚定坐标系**下计算 —— 与力场用同一套锚定（事件里已移植），
         这样机器人整体漂移不会污染"有没有让位到该让的地方"这个判断；
      3. σ=0.1 只针对单点误差。

    返回 ``(rows, cols, target, active)``；无活跃力场时 rows 为空。
    """
    from isaaclab.utils.math import quat_mul, quat_rotate

    fb = getattr(env, "_softsonic_active_force_body", None)
    if fb is None:
        return None
    active = fb >= 0
    if not active.any():
        return None
    rows = active.nonzero(as_tuple=False).squeeze(-1)
    cols = fb[rows]

    command = env.command_manager.get_term(command_name)
    ml = command.motion_lib
    total = ml.get_time_step_total(command.motion_ids)
    steps = torch.clamp(
        command.motion_start_time_steps + command.time_steps,
        torch.zeros_like(total), total - 1,
    )
    a_rot = env._softsonic_ff_anchor_rot[rows]        # noqa: SLF001
    a_pos = env._softsonic_ff_anchor_pos[rows]        # noqa: SLF001
    a_ref = env._softsonic_ff_anchor_ref_pos[rows]    # noqa: SLF001

    tgt_pos = (
        ml.get_body_pos_w_aug_full(command.motion_ids, steps)[rows, cols]
        + env.scene.env_origins[rows]
    )
    anchored = quat_rotate(a_rot, tgt_pos - a_ref) + a_pos
    anchored[:, 2] = tgt_pos[:, 2]                    # 只锚 XY+偏航，与力场一致
    if not want_quat:
        return rows, cols, anchored, active
    tgt_quat = ml.get_body_quat_w_aug_full(command.motion_ids, steps)[rows, cols]
    return rows, cols, (anchored, quat_mul(a_rot, tgt_quat)), active


def compliant_force_link_pos_error(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """受力连杆位置跟踪（锚定坐标系，对 q_aug）。SoftMimic 权重 3.0 / σ=0.1。"""
    out = _force_link_anchored_target(env, command_name)
    reward = torch.ones(env.num_envs, device=env.device)
    if out is None:
        return reward
    rows, cols, target, _ = out
    cur = env.scene[asset_cfg.name].data.body_pos_w[rows, cols]
    err = (target - cur).norm(dim=-1)
    reward[rows] = torch.exp(-(err * err) / (std * std))
    return reward


def compliant_force_link_ori_error(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """受力连杆朝向跟踪（锚定坐标系，对 q_aug）。SoftMimic 权重 3.0 / σ=0.1。"""
    from isaaclab.utils.math import quat_error_magnitude

    out = _force_link_anchored_target(env, command_name, want_quat=True)
    reward = torch.ones(env.num_envs, device=env.device)
    if out is None:
        return reward
    rows, cols, (_, tgt_quat), _ = out
    cur = env.scene[asset_cfg.name].data.body_quat_w[rows, cols]
    err = quat_error_magnitude(tgt_quat, cur)
    reward[rows] = torch.exp(-(err * err) / (std * std))
    return reward


def applied_force_tracking_error(env: ManagerBasedRLEnv, std: float = 20.0) -> torch.Tensor:
    """交互力跟踪奖励：实际力应接近期望力（SoftMimic Table VI，sigma 20 N，权重 2.0）。

    这一项只有在**力场**设定下才有意义：力由 ``F = k_ff·(设定点 - 手位)`` 决定，
    机器人靠自身姿态影响它。若改成逐帧回放记录的力，实际力恒等于期望力，本项退化
    成常数、梯度为零。

    无活跃力场的环境返回 1.0（满分），这样该项不会在休息期产生任何梯度。

    依赖 ``apply_softsonic_force_field`` 事件项写入的缓冲。IsaacLab 的管理器顺序是
    终止 -> 奖励 -> 指令推进 -> 事件，所以奖励读到的正是本步物理施加的力，无滞后。
    """
    buffers = _softsonic_wrench_buffers(env, "force")
    if buffers is None:   # 首步，事件项尚未跑过
        return torch.ones(env.num_envs, device=env.device)
    actual, desired, active = buffers
    err = (actual - desired).norm(dim=-1)
    reward = torch.exp(-(err * err) / (std * std))
    return torch.where(active, reward, torch.ones_like(reward))


def applied_torque_tracking_error(env: ManagerBasedRLEnv, std: float = 2.0) -> torch.Tensor:
    """交互力矩跟踪奖励（SoftMimic Table VI，sigma 2 N·m，权重 2.0）。"""
    buffers = _softsonic_wrench_buffers(env, "torque")
    if buffers is None:   # 首步，事件项尚未跑过
        return torch.ones(env.num_envs, device=env.device)
    actual, desired, active = buffers
    err = (actual - desired).norm(dim=-1)
    reward = torch.exp(-(err * err) / (std * std))
    return torch.where(active, reward, torch.ones_like(reward))


def alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """存活奖励（SoftMimic Table VI 权重 1.5）。

    SONIC 发布配置的 12 个奖励项里没有存活项。平时无所谓，但我们把终止判据改瞄
    q_aug、放宽阈值之后，探索期需要一个"活着就有分"的托底项，否则策略容易学成
    提前终止来规避负项。
    """
    return torch.ones(env.num_envs, device=env.device)
