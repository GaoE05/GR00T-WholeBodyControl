"""Execute actual source functions without Isaac Sim; no CUDA allocation.

Simulator imports are stubbed; quaternion math comes from unmodified IsaacLab.
"""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
BASELINE = '97f5dea8973a1135ce2e6946e63f80cdb3fe0574'
spec = importlib.util.spec_from_file_location('softsonic_test_math', '/workspace/isaaclab/source/isaaclab/isaaclab/utils/math.py')
math = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = math
spec.loader.exec_module(math)


def module(name, **attrs):
    mod = ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


module('isaaclab')
module('isaaclab.utils', math=math)
sys.modules['isaaclab.utils.math'] = math
module('isaaclab.assets', Articulation=object)
module('gear_sonic')
module('gear_sonic.trl')
module('gear_sonic.utils')


def source(path, revision=None):
    if revision:
        return subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT, text=True)
    return (ROOT / path).read_text()


def functions(path, names, revision=None, **extra):
    tree = ast.parse(source(path, revision))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for n in nodes:
        n.decorator_list = []
    ns = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
    ns.update(SceneEntityCfg=lambda name: NS(name=name), _get_body_indexes=lambda c, b: slice(None))
    ns.update(extra)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), 'exec'), ns)
    return NS(**{name: ns[name] for name in names})


heading = functions('gear_sonic/trl/utils/torch_transform.py', ['get_heading_q'])
module('gear_sonic.trl.utils', torch_transform=heading)


def quat(roll=0., pitch=0., yaw=0.):
    return math.quat_from_euler_xyz(*(torch.tensor([v], dtype=torch.float64) for v in (roll, pitch, yaw)))


def close(a, b, tol=1e-9):
    torch.testing.assert_close(a, torch.as_tensor(b, dtype=a.dtype).expand_as(a), rtol=0, atol=tol)


def forcefield_fixture():
    """Tensor motion buffers and a capturing physics sink for the real event."""
    tree = ast.parse(source('gear_sonic/utils/motion_lib/motion_lib_base.py'))
    names = {'SOFTSONIC_FORCE_BODIES', 'SOFTSONIC_SLICES'}
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    ns = {}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), '<motion constants>', 'exec'), ns)
    mlb = module('gear_sonic.utils.motion_lib.motion_lib_base', **{n: ns[n] for n in names})
    module('gear_sonic.utils.motion_lib', motion_lib_base=mlb)
    n = len(mlb.SOFTSONIC_FORCE_BODIES)
    ref_pos = torch.zeros(1, n, 3)
    ref_pos[:, :, 2] = 1.
    ref_quat = quat().float()[:, None].repeat(1, n, 1)
    aug_pos, aug_quat = ref_pos.clone(), ref_quat.clone()
    ss = torch.zeros(1, 15)
    ss[:, 7] = 100.
    lib = NS(body_pos_w_full=ref_pos, has_aug_pose=True,
             get_time_step_total=lambda ids: torch.full_like(ids, 100),
             get_motion_softsonic=lambda ids, steps: ss,
             get_body_pos_w_full=lambda ids, steps: ref_pos,
             get_body_quat_w_full=lambda ids, steps: ref_quat,
             get_body_pos_w_aug_full=lambda ids, steps: aug_pos,
             get_body_quat_w_aug_full=lambda ids, steps: aug_quat)
    c = NS(motion_lib=lib, motion_ids=torch.zeros(1, dtype=torch.long),
           motion_start_time_steps=torch.zeros(1, dtype=torch.long),
           time_steps=torch.zeros(1, dtype=torch.long), cfg=NS(reset_from_compliant_target=False))
    captured = {}
    robot = NS(num_bodies=n, body_names=mlb.SOFTSONIC_FORCE_BODIES,
               data=NS(root_pos_w=ref_pos[:, 0].clone(), root_quat_w=quat().float(),
                       body_pos_w=ref_pos.clone(), body_quat_w=ref_quat.clone()),
               permanent_wrench_composer=NS(set_forces_and_torques=lambda **kwargs: captured.update(kwargs)),
               set_softsonic_world_wrench=lambda forces, torques: captured.update(forces=forces, torques=torques))

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    env = NS(scene=Scene(robot=robot), command_manager=NS(get_term=lambda _: c),
             num_envs=1, device='cpu', episode_length_buf=torch.zeros(1, dtype=torch.long))
    return NS(env=env, command=c, robot=robot, ss=ss, ref_pos=ref_pos, ref_quat=ref_quat,
              aug_pos=aug_pos, aug_quat=aug_quat, captured=captured)
