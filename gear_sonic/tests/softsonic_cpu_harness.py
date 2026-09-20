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
