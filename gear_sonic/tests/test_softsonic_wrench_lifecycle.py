"""B1/N3 CPU regression with the REAL IsaacLab composer and write/reset methods.

Only the PhysX sink/actuators are replaced. No simulator, app, or CUDA context.
"""
import ast
import importlib.util
import math as pymath
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch
import warp as wp
import softsonic_cpu_harness as h

LAB = Path('/workspace/isaaclab/source/isaaclab/isaaclab')
wp.config.kernel_cache_dir = '/tmp/softsonic-wrench-cpu-warp'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


h.module('isaaclab.utils.warp')
load('isaaclab.utils.warp.kernels', LAB / 'utils/warp/kernels.py')
Composer = load('softsonic_real_composer', LAB / 'utils/wrench_composer.py').WrenchComposer
# Compile the actual upstream methods into a simulation-free base class.
tree = ast.parse((LAB / 'assets/articulation/articulation.py').read_text())
original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Articulation')
base = ast.ClassDef(name='Base', bases=[], keywords=[], decorator_list=[],
                    body=[n for n in original.body if isinstance(n, ast.FunctionDef)
                          and n.name in ('reset', 'write_data_to_sim')])
ns = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), base], type_ignores=[])), '<real articulation methods>', 'exec'), ns)
Base = ns['Base']
Base.instantaneous_wrench_composer = property(lambda r: r._instantaneous_wrench_composer)
Base.permanent_wrench_composer = property(lambda r: r._permanent_wrench_composer)
sys.modules['isaaclab.assets'].Articulation = Base
Robot = load('softsonic_articulation_test', h.ROOT / 'gear_sonic/envs/manager_env/softsonic_articulation.py').SoftSONICArticulation


def fixture():
    r = Robot()
    r.num_instances, r.num_bodies, r.device = 2, 2, 'cpu'
    q = h.quat().float()[:, None].repeat(2, 2, 1)
    r.data = NS(body_link_pos_w=torch.zeros(2, 2, 3), body_link_quat_w=q)
    r._instantaneous_wrench_composer = Composer(r)
    r._permanent_wrench_composer = Composer(r)
    r._ALL_INDICES = torch.arange(2, dtype=torch.int32)
    r._ALL_INDICES_WP = wp.from_torch(r._ALL_INDICES, dtype=wp.int32)
    r._ALL_BODY_INDICES_WP = wp.from_torch(torch.arange(2, dtype=torch.int32), dtype=wp.int32)
    r.actuators, r._has_implicit_actuators = {}, False
    r._apply_actuator_model = lambda: None
    r._joint_effort_target_sim = torch.zeros(2, 2)
    submitted = []
    def sink(**kw):
        assert kw['is_global'] is False
        q = r.data.body_link_quat_w
        submitted.append((h.math.quat_apply(q, kw['force_data'].reshape(2, 2, 3)).clone(),
                          h.math.quat_apply(q, kw['torque_data'].reshape(2, 2, 3)).clone()))
    r.root_physx_view = NS(apply_forces_and_torques_at_position=sink,
                          set_dof_actuation_forces=lambda *a: None)
    return r, submitted


for decimation in (1, 4):
    r, submitted = fixture()
    f, t = torch.zeros(2, 2, 3), torch.zeros(2, 2, 3)
    f[0, 0], f[1, 1] = torch.tensor([10., 0., 2.]), torch.tensor([0., 20., 1.])
    t[:, :, 0] = .5
    r.set_softsonic_world_wrench(f, t)
    for step in range(3 * decimation):
        yaw = (step % 3) * pymath.pi / 4
        r.data.body_link_quat_w[:] = h.quat(roll=.1*step, yaw=yaw).float()[:, None]
        r.write_data_to_sim()
        torch.testing.assert_close(submitted[-1][0], f, rtol=0, atol=1e-5)
        torch.testing.assert_close(submitted[-1][1], t, rtol=0, atol=1e-6)
        assert not r.instantaneous_wrench_composer.active
        assert r.instantaneous_wrench_composer.composed_force_as_torch.count_nonzero() == 0
    # Asynchronous reset: only env0 stops; env1 holds the same world vector.
    r.reset(torch.tensor([0]))
    f[0] = 0; t[0] = 0
    r.data.body_link_quat_w[0] = h.quat(yaw=pymath.pi).float()
    r.write_data_to_sim()
    torch.testing.assert_close(submitted[-1][0], f, rtol=0, atol=1e-5)
    torch.testing.assert_close(submitted[-1][1], t, rtol=0, atol=1e-6)
    # Changed body clears prior body; subsequent no-force event is physically 0.
    f[:] = 0; t[:] = 0; f[1, 0, 2] = 7
    r.set_softsonic_world_wrench(f, t); r.write_data_to_sim()
    torch.testing.assert_close(submitted[-1][0], f, rtol=0, atol=1e-5)
    f[:] = 0
    r.set_softsonic_world_wrench(f, t)
    for _ in range(decimation):
        r.write_data_to_sim()
        assert submitted[-1][0].count_nonzero() == 0
        assert submitted[-1][1].count_nonzero() == 0
    assert r.permanent_wrench_composer.composed_force_as_torch.count_nonzero() == 0
    assert r.permanent_wrench_composer.composed_torque_as_torch.count_nonzero() == 0
print('B1 actual composer + upstream write/reset PASS: decimation=1/4, rotations, async reset, body change, zero-force')

# Demonstrate the old failure through the exact same real composer.
r, submitted = fixture()
f = torch.zeros(2, 2, 3); f[:, 0, 0] = 10
r.permanent_wrench_composer.set_forces_and_torques(forces=f, is_global=True)
r.data.body_link_quat_w[:] = h.quat(yaw=pymath.pi/2).float()[:, None]
r.permanent_wrench_composer.set_forces_and_torques(forces=f, is_global=True)
Base.write_data_to_sim(r)
assert torch.linalg.vector_norm(submitted[-1][0] - f, dim=-1).max() > 14
print('B1 old cached-direction counterexample reproduced (>14 N vector error)')

# Event-specific regressions execute the production function.
event = h.functions('gear_sonic/envs/manager_env/mdp/events.py', ['apply_softsonic_force_field']).apply_softsonic_force_field
x = h.forcefield_fixture()
x.ss[:, 9] = .2; x.ss[:, 8] = 5; x.ss[:, 12] = .1
event(x.env, None, max_force=0, max_torque=0)
assert torch.isfinite(x.env._softsonic_force_actual).all()
assert x.captured['forces'].count_nonzero() == x.captured['torques'].count_nonzero() == 0
for limit in (-1., float('nan'), float('inf')):
    try:
        event(x.env, None, max_force=limit)
    except ValueError:
        pass
    else:
        raise AssertionError(limit)
# Pure torque and centered zero desired-force spring remain live.
x.ss[:, 7] = 0; x.ss[:, 8] = 5
event(x.env, None)
assert x.env._softsonic_active.all() and x.captured['torques'].norm() > 0
x.ss[:, 7] = 100; x.ss[:, 8] = 0; x.ss[:, 9:15] = 0
x.robot.data.body_pos_w[:, 0, 0] += .1
event(x.env, None)
assert x.captured['forces'].norm() > 9
# A body switch changes the anchor, rather than inheriting the old one.
x.ss[:, 6] = 1; x.robot.data.root_pos_w[:, 0] = .3
event(x.env, None)
h.close(x.env._softsonic_ff_anchor_pos, x.robot.data.root_pos_w)
# Missing metadata is a clear failure AND clears the owned held wrench.
x.command.motion_lib.has_softsonic = False
try:
    event(x.env, None)
except RuntimeError as exc:
    assert 'metadata' in str(exc)
else:
    raise AssertionError('missing metadata silently accepted')
assert x.captured['forces'].count_nonzero() == x.captured['torques'].count_nonzero() == 0
assert x.env._softsonic_force_actual.count_nonzero() == 0
print('N3 bounds + pure torque + centered spring + body reanchor + missing-metadata fail-closed PASS')
