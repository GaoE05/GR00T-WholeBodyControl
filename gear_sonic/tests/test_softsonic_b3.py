"""B3: execute the old/new forcefield with noncommuting yaw/roll rotations."""
import math as pymath
import torch
from softsonic_cpu_harness import BASELINE, close, forcefield_fixture, functions, math, quat

PATH = 'gear_sonic/envs/manager_env/mdp/events.py'


def run(revision):
    x = forcefield_fixture()
    A, D = quat(yaw=pymath.pi / 2).float(), quat(roll=.2).float()
    x.robot.data.root_quat_w = A
    x.ss[:, :3] = torch.tensor([[10., 0., 0.]])
    x.ss[:, 3:6] = torch.tensor([[.1, 0., 0.]])
    x.ss[:, 8] = 1.
    x.ss[:, 9:12] = torch.tensor([[.2, 0., 0.]])
    x.ss[:, 12:15] = torch.tensor([[.2, 0., 0.]])
    x.robot.data.body_pos_w[:, 0] += torch.tensor([[0., .1, 0.]])
    x.robot.data.body_quat_w[:, 0] = math.quat_mul(A, D)
    functions(PATH, ['apply_softsonic_force_field'], revision).apply_softsonic_force_field(x.env, None)
    return x


old, new = run(BASELINE), run(None)
old_angle = old.env._softsonic_torque_actual.norm()
new_angle = new.env._softsonic_torque_actual.norm()
close(new_angle, 0., 1e-6)
assert .282 < old_angle < .283
close(new.env._softsonic_force_desired, [[0., 10., 0.]], 2e-6)
close(new.env._softsonic_torque_desired, [[0., .1, 0.]], 1e-6)
old_force_error = (old.env._softsonic_force_actual - old.env._softsonic_force_desired).norm()
new_force_error = (new.env._softsonic_force_actual - new.env._softsonic_force_desired).norm()
close(new_force_error, 0., 3e-6)
assert 14.14 < old_force_error < 14.15
print('B3 setpoint rotation error rad before/after:', old_angle.item(), new_angle.item())
print('B3 force target mismatch N before/after:', old_force_error.item(), new_force_error.item())
print('B3 desired torque before/after:', old.env._softsonic_torque_desired.tolist(), new.env._softsonic_torque_desired.tolist())
print('B3 PASS')
