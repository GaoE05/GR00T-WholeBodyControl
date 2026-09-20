"""B5: adapted reset needs adapted anchors; q_ref reset retains its geometry."""
import torch
from softsonic_cpu_harness import BASELINE, close, forcefield_fixture, functions, math, quat
EVENTS = 'gear_sonic/envs/manager_env/mdp/events.py'

def case(aug_reset, revision=None, yaw=0.):
    x = forcefield_fixture()
    x.command.cfg.reset_from_compliant_target = aug_reset
    x.aug_pos[:, :, 0] += .1
    x.aug_quat[:] = quat(yaw=yaw).float()[:,None]
    x.ss[:, 9] = .2  # force-field setpoint displacement in world x
    if aug_reset:
        x.robot.data.root_pos_w[:] = x.aug_pos[:,0]
        x.robot.data.root_quat_w[:] = x.aug_quat[:,0]
        x.robot.data.body_pos_w[:] = x.aug_pos
        x.robot.data.body_quat_w[:] = x.aug_quat
    event = functions(EVENTS, ['apply_softsonic_force_field'], revision).apply_softsonic_force_field
    event(x.env, None)
    return x
old, new = case(True, BASELINE), case(True)
# Intended original world field O_ref=.2, body_aug=.1: force is 100*(.2-.1)=10N.
# The old q_ref root incorrectly translates the field by another .1m: 20N.
close(old.env._softsonic_force_actual[:,0], [20.], 1e-5)
close(new.env._softsonic_force_actual[:,0], [10.], 1e-5)
close(new.env._softsonic_ff_anchor_ref_pos, new.aug_pos[:,0])
print('B5 q_aug reset force N before/after:', old.env._softsonic_force_actual.tolist(), new.env._softsonic_force_actual.tolist())
# Both modes preserve their own root convention, also with different adapted yaw.
for mode in (False, True):
    x = case(mode, yaw=.4)
    expected_pos = x.aug_pos[:,0] if mode else x.ref_pos[:,0]
    close(x.env._softsonic_ff_anchor_ref_pos, expected_pos)
    close(x.env._softsonic_ff_anchor_rot, quat().float(), 1e-6)
# Default q_ref reset's field and wrench are unchanged by this root choice.
x, y = case(False, BASELINE), case(False)
close(x.env._softsonic_force_actual, y.env._softsonic_force_actual, 1e-6)
close(x.env._softsonic_force_setpoint, y.env._softsonic_force_setpoint, 1e-6)
print('B5 root mode PASS: q_aug translation/yaw consistent; q_ref geometry unchanged')
