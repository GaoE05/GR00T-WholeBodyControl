"""Event functions for domain randomization and environment resets in RL training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = None
    add_joint_default_pos = None
    add_hand_joint_default_pos = None
    base_com = None

    # interval - balance training
    push_robot = None

    randomize_rigid_body_mass = None

    # interval - SoftSONIC：从增强数据回放外力（见 replay_softsonic_external_wrench）
    softsonic_replay_wrench = None


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize joint default positions to simulate calibration errors.

    Applies random offsets to the default joint positions of the robot, modeling
    real-world joint encoder calibration inaccuracies. Also updates the action
    manager offset to keep action space aligned with the new defaults.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        asset_cfg: Scene entity config with joint IDs to randomize.
        pos_distribution_params: Min/max range for the position offset distribution.
        operation: How to combine the random value with the original ("add", "scale", "abs").
        distribution: Sampling distribution type.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos,
            pos_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically

        action_joint_names = env.action_manager.get_term("joint_pos")._joint_names
        asset_joint_names = asset.joint_names
        shared_joint_names = list(set(action_joint_names).intersection(set(asset_joint_names)))
        shared_joint_indices_action = [
            action_joint_names.index(name) for name in shared_joint_names
        ]
        shared_joint_indices_asset = [asset_joint_names.index(name) for name in shared_joint_names]

        shared_offset = asset.data.default_joint_pos[env_ids, shared_joint_indices_asset]
        env.action_manager.get_term("joint_pos")._offset[
            env_ids, shared_joint_indices_action
        ] = shared_offset


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu"
    ).unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


# ══════════════════════════════════════════════════════════════════════════════
# SoftSONIC：从增强数据回放外力
# ══════════════════════════════════════════════════════════════════════════════

def replay_softsonic_external_wrench(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "motion",
    force_threshold: float = 1.0,
    debug_print_every_n_steps: int = 0,
) -> None:
    """把 CMA 增强数据里记录的外力逐帧回放到仿真中。

    ## 为什么是回放而不是在仿真里采样

    ``q_aug`` 是 CMA 针对**某个特定力剖面**离线用 IK 解出来的。若仿真施加的力与
    生成时的不一致，``q_aug`` 就不是正确的监督目标。所以力必须随数据走 —— 这也是
    SoftMimic 的 ``(q_ref, w_i, K_cmd, q_aug)`` 四元组数据的含义。
    逐帧力数据由 ``scripts/cma_to_motionlib.py --with-softsonic`` 写入 motion_lib 的
    ``softsonic`` 字段，见 ``motion_lib_base.SOFTSONIC_SLICES``。

    ## 为什么用 interval 事件而不是 command term

    IsaacLab 的 ``ManagerBasedRLEnv.step`` 顺序是：
    ``apply_action -> write_data_to_sim -> sim.step``（decimation 循环内）
    ``-> command_manager.compute -> event_manager.apply(mode="interval")``。
    物理步进之前没有可用钩子（除 action manager），所以任何写入都会在**下一个**
    物理步生效。而 interval 事件跑在 ``command_manager.compute`` 之后 —— 那时
    ``TrackingCommand.time_steps`` 已经推进到 t+1，于是"读当前帧的力、写缓冲、
    下个物理步生效"时序自动对齐，**不需要手动补偿一帧**。

    配置上须设 ``interval_range_s=[step_dt, step_dt]`` 且 ``is_global_time=True``，
    这样每步触发一次、``env_ids`` 传 None（全部环境）。

    ## 两个容易搞错的地方

    - **``is_global=True`` 必须显式传**。CMA 存的力是世界系的（其可视化直接用
      连杆世界位置加力向量画箭头），而 IsaacLab 的接口默认是连杆局部系，不传的话
      力会跟着连杆一起转。
    - **判断是否施力要用 ``‖F‖``，不能用 ``force_body_id >= 0``**。CMA 存的
      link_id 取自 ``current_event or event_queue[0]``，含尚未开始的排队事件 ——
      实测 99% 的帧 id>=0 但只有约 66% 的帧 ‖F‖>1N。

    Args:
        env: 环境。
        env_ids: interval + is_global_time=True 时为 None，表示全部环境。
        asset_cfg: 机器人实体。
        command_name: 运动跟踪指令项的名字，用它拿 motion_ids 与 time_steps。
        force_threshold: ‖F‖ 低于此值（N）视为无外力，清零该环境的力。
        debug_print_every_n_steps: 非零时每 N 步打印一次施力统计，用于确认外力
            真的进了仿真（仅靠配置加载成功不足以证明这件事）。
    """
    from isaaclab.assets import Articulation as _Articulation
    from gear_sonic.utils.motion_lib import motion_lib_base as _mlb

    robot: _Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    motion_lib = command.motion_lib

    # 规范索引 -> 本资产的 body 索引。只解析一次，缓存在 env 上。
    cache_key = f"_softsonic_force_body_ids_{asset_cfg.name}"
    body_id_lut = getattr(env, cache_key, None)
    if body_id_lut is None:
        ids = []
        for name in _mlb.SOFTSONIC_FORCE_BODIES:
            if name not in robot.body_names:
                raise ValueError(
                    f"SOFTSONIC_FORCE_BODIES 里的 {name!r} 不在资产 "
                    f"{asset_cfg.name!r} 的 body 列表里。可用: {robot.body_names}"
                )
            ids.append(robot.body_names.index(name))
        body_id_lut = torch.tensor(ids, dtype=torch.long, device=env.device)
        setattr(env, cache_key, body_id_lut)

    # 绝对帧号是 motion_start_time_steps + time_steps。动作可能从随机帧开始
    # （TrackingCommand._resample_command 里 sample_time_steps），只用 time_steps
    # 会让回放的力与 q_aug 错位 —— 评估时起始帧恒为 0 所以看不出来，训练时才炸。
    # 另外在触发终止与 reset 之间该值可能刚好等于总帧数，夹一下避免越界。
    total = motion_lib.get_time_step_total(command.motion_ids)
    steps = torch.clamp(
        command.motion_start_time_steps + command.time_steps,
        torch.zeros_like(total),
        total - 1,
    )

    ss = motion_lib.get_motion_softsonic(command.motion_ids, steps)
    sl = _mlb.SOFTSONIC_SLICES
    force_w = ss[:, sl["ext_force"]]
    torque_w = ss[:, sl["ext_torque"]]
    canon = ss[:, sl["force_body_id"]].squeeze(-1).long()

    active = (torch.linalg.norm(force_w, dim=-1) > force_threshold) & (canon >= 0)

    # 全零缓冲，只把活跃环境的目标连杆填上。外力缓冲是持久的，所以每步都必须
    # 重写全部环境，否则无力的环境会残留上一帧的力。
    forces = torch.zeros(env.num_envs, robot.num_bodies, 3, device=env.device)
    torques = torch.zeros_like(forces)
    if active.any():
        rows = active.nonzero(as_tuple=False).squeeze(-1)
        cols = body_id_lut[canon[rows]]
        forces[rows, cols] = force_w[rows]
        torques[rows, cols] = torque_w[rows]

    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torques, body_ids=None, env_ids=None, is_global=True
    )

    if debug_print_every_n_steps:
        counter = getattr(env, "_softsonic_wrench_step", 0) + 1
        setattr(env, "_softsonic_wrench_step", counter)
        if counter % debug_print_every_n_steps == 0:
            mags = torch.linalg.norm(force_w[active], dim=-1) if active.any() else force_w.new_zeros(1)
            print(  # noqa: T201
                f"[softsonic wrench] step {counter}: 施力环境 "
                f"{int(active.sum())}/{env.num_envs}，‖F‖ 均值 {mags.mean():.2f} N "
                f"最大 {mags.max():.2f} N"
            )
