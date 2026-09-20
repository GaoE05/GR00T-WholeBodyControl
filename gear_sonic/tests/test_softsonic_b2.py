"""B2 numerical regressions. Run directly with IsaacLab python.sh on CPU."""
from types import SimpleNamespace as NS
import torch
from softsonic_cpu_harness import BASELINE, close, functions, math, quat

PATH = 'gear_sonic/envs/manager_env/mdp/rewards.py'
NAMES = ['_compliant_relative_ref', 'tracking_compliant_relative_body_pos_error',
         'tracking_compliant_relative_body_ori_error', 'tracking_compliant_local_vr_5point_error']


def scenario():
    ref = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    aug = ref + torch.tensor([[.1, 0., 0.]], dtype=torch.float64)
    offsets = torch.tensor([[[0., 0., 0.], [.3, .2, .4], [-.2, .1, -.5]]], dtype=torch.float64)
    c = NS(cfg=NS(body_names=['root', 'hand', 'foot'], reward_point_body=['root', 'hand', 'foot']),
           anchor_pos_w=ref, anchor_pos_w_aug=aug, anchor_quat_w=quat(), anchor_quat_w_aug=quat(),
           robot_anchor_pos_w=aug.clone(), robot_anchor_quat_w=quat(),
           body_pos_w_aug=aug[:, None] + offsets, body_quat_w_aug=quat()[:, None].repeat(1, 3, 1))
    c.robot_body_pos_w = c.body_pos_w_aug.clone()
    c.robot_body_quat_w = c.body_quat_w_aug.clone()
    c.reward_point_body_pos_w_aug = c.body_pos_w_aug.clone()
    c.robot_reward_point_body_pos_w = c.body_pos_w_aug.clone()
    return c, NS(num_envs=1, command_manager=NS(get_term=lambda _: c))


def scores(fns, env):
    return torch.stack([fns.tracking_compliant_relative_body_pos_error(env, 'motion', .1),
                        fns.tracking_compliant_relative_body_ori_error(env, 'motion', .4),
                        fns.tracking_compliant_local_vr_5point_error(env, 'motion', .06)])


old = functions(PATH, NAMES, BASELINE)
new = functions(PATH, NAMES + ['_compliant_yaw_alignment'])
c, env = scenario()
before, after = scores(old, env), scores(new, env)
close(after, 1.)
assert before[0] < .368 and before[2] < .063
print('B2 translation rewards before/after:', before.flatten().tolist(), after.flatten().tolist())
c.anchor_quat_w_aug = quat(.2, -.3, .5)
c.robot_anchor_quat_w = c.anchor_quat_w_aug.clone()
c.body_quat_w_aug = c.anchor_quat_w_aug[:, None].repeat(1, 3, 1)
c.robot_body_quat_w = c.body_quat_w_aug.clone()
close(scores(new, env), 1.)
print('B2 exact tilted q_aug before/after:', scores(old, env).flatten().tolist(), scores(new, env).flatten().tolist())
# XY/yaw drift is ignored; absolute height is still supervised, per upstream.
A = quat(yaw=.7)
c.robot_anchor_pos_w += torch.tensor([[.4, -.3, 0.]])
for attr in ['robot_body_pos_w', 'robot_reward_point_body_pos_w']:
    original = getattr(c, attr) - c.anchor_pos_w_aug[:, None]
    setattr(c, attr, math.quat_apply(A[:, None].repeat(1, 3, 1), original) + c.robot_anchor_pos_w[:, None])
c.robot_anchor_quat_w = math.quat_mul(A, c.anchor_quat_w_aug)
c.robot_body_quat_w = math.quat_mul(A[:, None].repeat(1, 3, 1), c.body_quat_w_aug)
close(scores(new, env), 1.)
c.robot_anchor_pos_w[:, 2] += .1
c.robot_body_pos_w[:, :, 2] += .1
c.robot_reward_point_body_pos_w[:, :, 2] += .1
height = scores(new, env)
assert height[0] < .368 and height[2] < .063
print('B2 +0.1m height still penalized:', height.flatten().tolist())
print('B2 PASS')
