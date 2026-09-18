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

    # interval - SoftSONIC：按增强数据的力场参数施加外力（见 apply_softsonic_force_field）
    softsonic_force_field = None
    # interval - SoftSONIC：运动学驱动，出 target motion 对比视频用
    # （见 softsonic_kinematic_drive）
    softsonic_kinematic_drive = None


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

def softsonic_kinematic_drive(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,  # noqa: ARG001
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "motion",
    use_aug: bool = True,
) -> None:
    """每步把机器人状态直接写成参考轨迹，用于出"target motion"那一路对比视频。

    ``use_aug=True`` 写柔顺监督目标 q_aug，即"一个完全柔顺的机器人此刻该在的位姿"；
    ``False`` 写原始参考 q_ref。这一路是**运动学驱动**，不受外力影响 —— 它给出的是
    对比视频里的"应该长什么样"，另外两路（frozen SONIC / SoftSONIC）才是物理仿真。

    为什么做成事件项而不用 ``run_replay``：后者的录像通路不可用（循环里只有
    ``sim.render()``、没有 ``sim.step()``，写进 PhysX 的状态从不刷到渲染管线，相机也
    停在原点）。做成逐步事件后，状态写入发生在正常的 ``env.step()`` 周期内，
    ``recorders=render`` 那条维护中的通路能正确取到画面。

    状态写入的四个量与 ``commands.py`` 的 reset 分支一致：只换关节角会留下不自洽的
    根位姿（q_aug 的骨盆相对 q_ref 中位移动 10.5cm、p90 达 26.4cm）。

    Args:
        asset_cfg: 机器人实体。
        command_name: 运动跟踪指令项的名字。
        use_aug: True 写 q_aug，False 写 q_ref。
    """
    from isaaclab.assets import Articulation as _Articulation

    robot: _Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    motion_lib = command.motion_lib

    total = motion_lib.get_time_step_total(command.motion_ids)
    steps = torch.clamp(
        command.motion_start_time_steps + command.time_steps,
        torch.zeros_like(total),
        total - 1,
    )

    if use_aug:
        if not motion_lib.has_aug_pose:
            raise RuntimeError(
                "use_aug=True 但动作数据里没有 pose_aa_aug。"
                "用 scripts/cma_to_motionlib.py --with-softsonic 重新生成。"
            )
        root_pos = motion_lib.get_body_pos_w_aug(command.motion_ids, steps)[:, 0]
        root_quat = motion_lib.get_body_quat_w_aug(command.motion_ids, steps)[:, 0]
        root_lin = motion_lib.get_body_lin_vel_w_aug(command.motion_ids, steps)[:, 0]
        root_ang = motion_lib.get_body_ang_vel_w_aug(command.motion_ids, steps)[:, 0]
        dof_pos = motion_lib.get_dof_pos_aug(command.motion_ids, steps)
        dof_vel = motion_lib.get_dof_vel_aug(command.motion_ids, steps)
    else:
        root_pos = motion_lib.get_root_pos_w(command.motion_ids, steps)
        root_quat = motion_lib.get_root_quat_w(command.motion_ids, steps)
        root_lin = motion_lib.get_root_lin_vel_w(command.motion_ids, steps)
        root_ang = motion_lib.get_root_ang_vel_w(command.motion_ids, steps)
        dof_pos = motion_lib.get_dof_pos(command.motion_ids, steps)
        dof_vel = motion_lib.get_dof_vel(command.motion_ids, steps)

    # 关节数可能多于动作数据的自由度（例如 43 DOF 机器人 + 29 DOF 数据），
    # 与 _update_replay_frame 相同的处理：按 body_joint_indices 映射，其余置零。
    n_robot = robot.num_joints
    if n_robot > dof_pos.shape[-1]:
        idx = getattr(env.wrapper, "_body_joint_indices", None)
        if idx is None:
            raise RuntimeError(f"机器人 {n_robot} 个关节多于数据的 {dof_pos.shape[-1]}，但缺少映射索引")
        full_pos = torch.zeros(env.num_envs, n_robot, device=env.device, dtype=dof_pos.dtype)
        full_vel = torch.zeros_like(full_pos)
        full_pos[:, idx] = dof_pos
        full_vel[:, idx] = dof_vel
        dof_pos, dof_vel = full_pos, full_vel

    robot.write_joint_state_to_sim(dof_pos, dof_vel)
    robot.write_root_state_to_sim(
        torch.cat([root_pos + env.scene.env_origins, root_quat, root_lin, root_ang], dim=-1)
    )


def apply_softsonic_force_field(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "motion",
    max_force: float = 140.0,
    max_torque: float = 10.0,
    debug_print_every_n_steps: int = 0,
) -> None:
    """按增强数据里的力场参数，在仿真中施加外力/力矩。

    ## 为什么是力场而不是直接回放记录的力

    直接回放 F(t) 会让**柔顺退让的机器人和硬扛的机器人感受到完全一样的力**，物理
    反馈回路消失：力跟踪奖励退化成与策略无关的常数，"柔顺降低交互力"这个安全性
    主张也无从度量。

    SoftMimic 的做法是模拟一个弹簧。设定点满足（``ik_update.py:202``）::

        p_ff = p_ref + F/k_ff + F/k_robot

    于是仿真里 ``F_actual = k_ff · (p_ff - p_hand)``::

        手到柔顺位置 p_des = p_ref + F/k_robot  ->  F_actual = F（期望值）
        手硬扛留在 p_ref                        ->  F_actual = F·(1 + k_ff/k_robot)
        手过度退让                              ->  F_actual 更小

    数据里存的是**相对参考手位的偏移** delta_ff = p_ff - p_ref（平移不变），
    运行时取 Isaac Lab 里参考连杆的世界位置再加上它。

    ## 活跃判据

    用 ``k_ff > 0``，不用 ``‖F‖`` 或 ``force_body_id >= 0``。上游 ``runner.py:427``
    里 ``ff_stiffness`` 初始为 0、只在 ``current_event`` 存在时才赋值，是"当前有事件
    正在施力"的精确标记；而 body id 含尚未开始的排队事件，力阈值会把斜坡起止段误判。

    ## 时序

    见 ``events.py`` 顶部关于 interval 事件的说明：本事件跑在
    ``command_manager.compute`` 之后，``time_steps`` 已推进到 t+1，写入的力在下一个
    物理步生效，时序自动对齐。

    Args:
        env: 环境。
        env_ids: interval + is_global_time=True 时为 None，表示全部环境。
        asset_cfg: 机器人实体。
        command_name: 运动跟踪指令项的名字。
        max_force: 力的幅值上限（N），与 CMA 的 ``IK_CHECK_MAX_FORCE_MAGNITUDE`` 一致，
            防止机器人被推飞时力发散。
        max_torque: 力矩幅值上限（N·m）。
        debug_print_every_n_steps: 非零时每 N 步打印期望力与实际力的对比，用于确认
            物理反馈确实生效（硬扛时实际力应大于期望力）。
    """
    from isaaclab.assets import Articulation as _Articulation
    from isaaclab.utils.math import (
        axis_angle_from_quat,
        euler_xyz_from_quat,
        quat_from_angle_axis,
        quat_from_euler_xyz,
        quat_inv,
        quat_mul,
        quat_rotate,
        wrap_to_pi,
    )

    from gear_sonic.utils.motion_lib import motion_lib_base as _mlb

    robot: _Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    motion_lib = command.motion_lib

    # 规范索引 -> 本资产的 body 索引，只解析一次。该索引同时用于
    # robot.data.body_pos_w 与 motion_lib 的 *_full 缓冲 —— 后者的文档写明是
    # "all bodies, IsaacLab order"，与 robot.body_names 同序。
    cache_key = f"_softsonic_force_body_ids_{asset_cfg.name}"
    body_id_lut = getattr(env, cache_key, None)
    if body_id_lut is None:
        n_full = motion_lib.body_pos_w_full.shape[1]
        if n_full != robot.num_bodies:
            raise ValueError(
                f"motion_lib 的 full body 数 {n_full} 与资产的 {robot.num_bodies} 不一致，"
                "无法共用同一套 body 索引。请检查 mujoco_to_isaaclab_body 配置。"
            )
        ids = []
        for name in _mlb.SOFTSONIC_FORCE_BODIES:
            if name not in robot.body_names:
                raise ValueError(
                    f"SOFTSONIC_FORCE_BODIES 里的 {name!r} 不在资产的 body 列表里"
                )
            ids.append(robot.body_names.index(name))
        body_id_lut = torch.tensor(ids, dtype=torch.long, device=env.device)
        setattr(env, cache_key, body_id_lut)

    # 绝对帧号 = motion_start_time_steps + time_steps（动作可能从随机帧开始）；
    # 在触发终止与 reset 之间该值可能等于总帧数，夹一下避免越界。
    total = motion_lib.get_time_step_total(command.motion_ids)
    steps = torch.clamp(
        command.motion_start_time_steps + command.time_steps,
        torch.zeros_like(total),
        total - 1,
    )

    ss = motion_lib.get_motion_softsonic(command.motion_ids, steps)
    sl = _mlb.SOFTSONIC_SLICES
    k_ff = ss[:, sl["ff_stiffness"]].squeeze(-1)
    k_ff_rot = ss[:, sl["ff_rot_stiffness"]].squeeze(-1)
    delta_p = ss[:, sl["ff_setpoint_delta_pos"]]
    delta_rv = ss[:, sl["ff_setpoint_delta_rotvec"]]
    canon = ss[:, sl["force_body_id"]].squeeze(-1).long()
    active = k_ff > 0.0

    forces = torch.zeros(env.num_envs, robot.num_bodies, 3, device=env.device)
    torques = torch.zeros_like(forces)

    # ── 力场锚定（移植自 SoftMimic） ─────────────────────────────────────────
    # compliance_augmented_reference_command.py:366-435 的
    # "anchor forcefield to the robot's state at the start of an interaction"。
    #
    # 为什么必须这么做：把设定点直接锚在**当前参考连杆位置**上，机器人根部相对参考
    # 的累积跟踪误差会原样变成额外的力。实测 error_anchor_pos ≈ 0.141m，乘 k_ff≈94
    # 约合 13.3N 的虚假力，而总 force_err 才 19.0N —— 大部分"额外的力"根本不是
    # 机器人硬扛出来的，是漂移造成的假象（这也是 force_ratio 4.26 越过刚性极限
    # 3.09 的原因）。
    #
    # SoftMimic 的做法是在力事件的**上升沿**记录机器人实际根位姿与参考根位姿，
    # 之后整个事件期间按 O_new = R_delta·(O_data - P_ref_start) + P_robot_start
    # 把力场几何搬到机器人所在的位置。只补偿平移与偏航、Z 保持数据原值。
    _ANCHOR_K_EPS = 0.1  # 与 SoftMimic 的 rising-edge 阈值一致
    last_k = getattr(env, "_softsonic_last_k_ff", None)
    if last_k is None or last_k.shape[0] != env.num_envs:
        last_k = torch.zeros(env.num_envs, device=env.device)
        env._softsonic_ff_anchor_pos = torch.zeros(env.num_envs, 3, device=env.device)  # noqa: SLF001
        env._softsonic_ff_anchor_ref_pos = torch.zeros(env.num_envs, 3, device=env.device)  # noqa: SLF001
        env._softsonic_ff_anchor_rot = torch.zeros(env.num_envs, 4, device=env.device)  # noqa: SLF001
        env._softsonic_ff_anchor_rot[:, 0] = 1.0  # noqa: SLF001  单位四元数

    # episode 刚 reset 的环境必须重新锚定，否则会沿用上一条 episode 的锚点。
    # 对应 SoftMimic 在 reset()/reset_motions() 里把 _last_ff_stiffness 清零。
    just_reset = env.episode_length_buf <= 1
    last_k = torch.where(just_reset, torch.zeros_like(last_k), last_k)

    ref_root_pos = (
        motion_lib.get_body_pos_w_full(command.motion_ids, steps)[:, 0]
        + env.scene.env_origins
    )
    ref_root_quat = motion_lib.get_body_quat_w_full(command.motion_ids, steps)[:, 0]

    rising = (last_k < _ANCHOR_K_EPS) & (k_ff >= _ANCHOR_K_EPS)
    if rising.any():
        ids = rising.nonzero(as_tuple=False).squeeze(-1)
        _, _, yaw_now = euler_xyz_from_quat(robot.data.root_quat_w[ids])
        _, _, yaw_ref = euler_xyz_from_quat(ref_root_quat[ids])
        d_yaw = wrap_to_pi(yaw_now - yaw_ref)
        zeros = torch.zeros_like(d_yaw)
        env._softsonic_ff_anchor_rot[ids] = quat_from_euler_xyz(zeros, zeros, d_yaw)  # noqa: SLF001
        env._softsonic_ff_anchor_pos[ids] = robot.data.root_pos_w[ids]  # noqa: SLF001
        env._softsonic_ff_anchor_ref_pos[ids] = ref_root_pos[ids]  # noqa: SLF001
    env._softsonic_last_k_ff = k_ff.clone()  # noqa: SLF001

    if active.any():
        rows = active.nonzero(as_tuple=False).squeeze(-1)
        cols = body_id_lut[canon[rows]]

        # 参考连杆位姿（Isaac Lab 世界系）。用 *_full 而非 body_pos_w：受力连杆未必
        # 在运动指令的 14 个 body_names 里（例如 shoulder_pitch 就不在）。
        ref_pos_full = motion_lib.get_body_pos_w_full(command.motion_ids, steps)
        ref_quat_full = motion_lib.get_body_quat_w_full(command.motion_ids, steps)
        ref_pos = ref_pos_full[rows, cols] + env.scene.env_origins[rows]
        ref_quat = ref_quat_full[rows, cols]

        # 施加锚定变换：把参考连杆位姿搬到"事件起始时机器人所在"的坐标系下。
        a_rot = env._softsonic_ff_anchor_rot[rows]  # noqa: SLF001
        a_pos = env._softsonic_ff_anchor_pos[rows]  # noqa: SLF001
        a_ref = env._softsonic_ff_anchor_ref_pos[rows]  # noqa: SLF001
        anchored = quat_rotate(a_rot, ref_pos - a_ref) + a_pos
        anchored[:, 2] = ref_pos[:, 2]   # 只锚 XY+偏航，Z 保持数据原值
        ref_pos = anchored
        ref_quat = quat_mul(a_rot, ref_quat)

        hand_pos = robot.data.body_pos_w[rows, cols]
        hand_quat = robot.data.body_quat_w[rows, cols]

        # 线性：F = k_ff · (设定点 - 实际手位)。偏移量也要跟着偏航一起转。
        setpoint_pos = ref_pos + quat_rotate(a_rot, delta_p[rows])
        f = k_ff[rows, None] * (setpoint_pos - hand_pos)

        # 旋转：设定点朝向 = Rot(delta_rv) · 参考朝向，力矩 = k_ff_rot · rotvec(误差)
        ang = torch.linalg.norm(delta_rv[rows], dim=-1)
        axis = torch.where(
            ang[:, None] > 1e-8, delta_rv[rows] / ang.clamp(min=1e-8)[:, None],
            torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(delta_rv[rows]),
        )
        setpoint_quat = quat_mul(quat_from_angle_axis(ang, axis), ref_quat)
        t = k_ff_rot[rows, None] * axis_angle_from_quat(
            quat_mul(setpoint_quat, quat_inv(hand_quat))
        )

        # 夹幅值，防止机器人被推飞后力发散
        f = f * (max_force / torch.linalg.norm(f, dim=-1, keepdim=True).clamp(min=max_force))
        t = t * (max_torque / torch.linalg.norm(t, dim=-1, keepdim=True).clamp(min=max_torque))

        forces[rows, cols] = f
        torques[rows, cols] = t
        env._softsonic_last_force = f  # noqa: SLF001  debug 用（只含活跃行）

    # 供力跟踪奖励读取的全尺寸缓冲。奖励在下一步的 line 208 计算，那时这里存的正是
    # 本步物理实际施加的力 —— IsaacLab 的顺序是 终止(204) -> 奖励(208) ->
    # 指令推进(232) -> 事件(235)，所以不存在滞后。
    actual_f = torch.zeros(env.num_envs, 3, device=env.device)
    actual_t = torch.zeros_like(actual_f)
    if active.any():
        actual_f[rows] = f
        actual_t[rows] = t
    env._softsonic_force_actual = actual_f  # noqa: SLF001
    env._softsonic_torque_actual = actual_t  # noqa: SLF001
    env._softsonic_force_desired = ss[:, sl["desired_force"]].clone()  # noqa: SLF001
    env._softsonic_torque_desired = ss[:, sl["desired_torque"]].clone()  # noqa: SLF001

    # 逐帧的"完全刚性"参考值，供评价指标归一化用。
    #
    # 必须逐帧算而不能用一个常数：k_ff 是逐事件对数均匀采样的（实测本数据集
    # 12.2~834.8），刚性极限 1+k_ff/k_robot 的范围是 1.27~19.55 —— 中位 3.09、
    # **均值 4.96**。拿均值形式的 force_ratio 去比中位常数会得出"比刚性还差"的
    # 错误结论。
    #
    # 推导：数据满足 dp = F/k_ff + F/k_robot（已用 MuJoCo 交叉验证，自洽误差 0.00%），
    # 两边乘 k_ff/F 即得
    #     k_ff·dp/F = 1 + k_ff/k_robot = 刚性极限
    # 于是完全不需要知道 k_robot，三个字段直接给出该帧的刚性参考值。
    want_n = ss[:, sl["desired_force"]].norm(dim=-1)
    dp_n = ss[:, sl["ff_setpoint_delta_pos"]].norm(dim=-1)
    env._softsonic_rigid_limit = torch.where(  # noqa: SLF001
        want_n > 1e-6, k_ff * dp_n / want_n.clamp(min=1e-6), torch.ones_like(want_n)
    )
    env._softsonic_active = active.clone()  # noqa: SLF001
    env._softsonic_k_ff = k_ff.clone()  # noqa: SLF001  critic 特权观测用

    # 力场设定点（世界系），供对比视频画"力往哪拉"。无力的环境填零。
    sp_buf = torch.zeros(env.num_envs, 3, device=env.device)
    if active.any():
        sp_buf[rows] = setpoint_pos
    env._softsonic_force_setpoint = sp_buf  # noqa: SLF001

    # 受力连杆在**本资产**里的 body 索引（无力的环境填 -1）。
    # 供 compliant_force_link_* 奖励使用 —— 对照 SoftMimic 的
    # force_link_keypoint_tracking_local，它同样只惩罚受力连杆那一个点的误差。
    fb = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    if active.any():
        fb[rows] = cols
    env._softsonic_active_force_body = fb  # noqa: SLF001

    # 外力缓冲是持久的，必须每步重写全部环境，否则无力环境残留上一帧
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torques, body_ids=None, env_ids=None, is_global=True
    )

    if debug_print_every_n_steps:
        counter = getattr(env, "_softsonic_wrench_step", 0) + 1
        env._softsonic_wrench_step = counter  # noqa: SLF001
        if counter % debug_print_every_n_steps == 0:
            if active.any():
                want = torch.linalg.norm(ss[active][:, sl["desired_force"]], dim=-1)
                got = torch.linalg.norm(env._softsonic_last_force, dim=-1)  # noqa: SLF001
                # 比值只在期望力足够大时才有意义：力斜坡起止段期望力接近 0，
                # 小分母会把比值放大到上百，是统计假象而非物理现象。
                sig = want > 1.0
                ratio = (
                    (got[sig] / want[sig]).mean().item() if sig.any() else float("nan")
                )
                # 锚定诊断：drift 是机器人根部相对参考的偏离，corr 是锚定实际
                # 搬动力场的距离。若 body 0 不是根连杆，drift 会是个离谱的大数。
                drift = (robot.data.root_pos_w - ref_root_pos).norm(dim=-1)
                corr = (
                    quat_rotate(
                        env._softsonic_ff_anchor_rot[active],  # noqa: SLF001
                        ref_root_pos[active] - env._softsonic_ff_anchor_ref_pos[active],  # noqa: SLF001
                    )
                    + env._softsonic_ff_anchor_pos[active]  # noqa: SLF001
                    - ref_root_pos[active]
                )[:, :2].norm(dim=-1)
                print(  # noqa: T201
                    f"[softsonic ff] step {counter}: 活跃 {int(active.sum())}/{env.num_envs}，"
                    f"期望力 {want.mean():.2f} N，实际力 {got.mean():.2f} N，"
                    f"比值 {ratio:.2f}（仅统计期望力>1N 的环境；"
                    "完全刚性的理论极限是 1 + k_ff/k_robot）"
                    f"；根部漂移 {drift.mean():.3f} m，锚定修正 {corr.mean():.3f} m"
                )
            else:
                print(f"[softsonic ff] step {counter}: 无活跃力场")  # noqa: T201
