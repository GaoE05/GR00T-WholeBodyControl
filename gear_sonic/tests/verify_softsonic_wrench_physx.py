#!/usr/bin/env python3
"""Short real-PhysX B1 acceptance; run only on an explicitly available GPU.

From fork root:
  python gear_sonic/tests/verify_softsonic_wrench_physx.py --headless --device cuda:0 --output /tmp/b1.json

Captures arguments at the actual PhysX submission boundary, reconstructs world
wrench using current link quaternions, then calls the original PhysX API. Both
decimation=1/4, asynchronous reset, two bodies, and zero-field are covered. This
is a force transport acceptance, not a policy/forcefield performance benchmark.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from filelock import FileLock
with FileLock('/tmp/isaaclab_app_launcher.lock'):
    launcher = AppLauncher(args)
app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG
from gear_sonic.envs.manager_env.softsonic_articulation import SoftSONICArticulation


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')


cfg = SceneCfg(num_envs=args.num_envs, env_spacing=3.0)
cfg.robot.class_type = SoftSONICArticulation
# Floating in free space avoids ground impulses and isolates wrench transport.
cfg.robot.spawn.rigid_props.disable_gravity = True
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=args.device))
scene = InteractiveScene(cfg)
sim.reset()
robot = scene['robot']
assert isinstance(robot, SoftSONICArticulation)
assert args.num_envs >= 2
n, b = robot.num_instances, robot.num_bodies
force, torque = torch.zeros(n, b, 3, device=args.device), torch.zeros(n, b, 3, device=args.device)
left = robot.body_names.index('left_wrist_yaw_link')
right = robot.body_names.index('right_wrist_yaw_link')
records = []
context = {}


class PhysXSubmissionSpy:
    def __init__(self, original):
        self.original = original
        self.previous_key = None
        self.previous_quat = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def apply_forces_and_torques_at_position(self, **kw):
        assert kw['is_global'] is False
        # Independent PhysX read: do not validate a stale ArticulationData cache
        # against itself. PhysX exposes xyzw whereas IsaacLab math uses wxyz.
        q_xyzw = self.original.get_link_transforms()[..., 3:7]
        q = q_xyzw[..., [3, 0, 1, 2]]
        key = (context['decimation'], context['control_step'])
        hold_rotation = 0.0
        if key == self.previous_key:
            dot = (q * self.previous_quat).sum(-1).abs().clamp(max=1.0)
            hold_rotation = (2 * torch.acos(dot)).max().item()
        self.previous_key, self.previous_quat = key, q.clone()
        f = quat_apply(q, kw['force_data'].reshape(n, b, 3))
        t = quat_apply(q, kw['torque_data'].reshape(n, b, 3))
        ferr = (f-force).norm(dim=-1).max().item()
        terr = (t-torque).norm(dim=-1).max().item()
        records.append(dict(context, within_hold_link_rotation_rad=hold_rotation,
                            max_force_vector_error_N=ferr,
                            max_torque_vector_error_Nm=terr,
                            submitted_force_norm_N=f.norm(dim=-1).max().item()))
        assert ferr < 1e-4 and terr < 1e-5, records[-1]
        return self.original.apply_forces_and_torques_at_position(**kw)


robot._root_physx_view = PhysXSubmissionSpy(robot.root_physx_view)
try:
    for decimation in (1, 4):
        for control_step in range(8):
            # Root reset includes 0/45/90 degree yaw and a nonzero yaw velocity,
            # so physical substeps see different orientations during each hold.
            if control_step == 0:
                state = robot.data.default_root_state.clone()
                state[:, :3] += scene.env_origins
                yaw = torch.arange(n, device=args.device).float().remainder(3) * (torch.pi/4)
                zeros = torch.zeros_like(yaw)
                state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
                state[:, 12] = .7
                robot.write_root_state_to_sim(state)
                robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
                robot.reset()
                scene.update(.005)
            force.zero_(); torque.zero_()
            selected = left if control_step < 3 else right
            if control_step < 6:
                force[:, selected] = torch.tensor([10., -3., 2.], device=args.device)
                torque[:, selected] = torch.tensor([.2, .1, -.1], device=args.device)
            robot.set_softsonic_world_wrench(force, torque)
            robot.set_joint_position_target(robot.data.default_joint_pos)
            for substep in range(decimation):
                if control_step == 4 and substep == 0:
                    # Other environments retain their force across env0 reset.
                    robot.reset(torch.tensor([0], device=args.device))
                    force[0].zero_(); torque[0].zero_()
                context.update(decimation=decimation, control_step=control_step, substep=substep)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(.005)
                assert robot.instantaneous_wrench_composer.composed_force_as_torch.count_nonzero() == 0
    assert len(records) == 40, len(records)
    assert any(r['decimation'] == 4 and r['within_hold_link_rotation_rad'] > 1e-5
               for r in records), 'test did not rotate a link within a held wrench interval'
    result = {'status': 'PASS', 'num_envs': n, 'physics_steps': len(records),
              'max_force_vector_error_N': max(r['max_force_vector_error_N'] for r in records),
              'max_torque_vector_error_Nm': max(r['max_torque_vector_error_Nm'] for r in records),
              'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}))
finally:
    app.close()
