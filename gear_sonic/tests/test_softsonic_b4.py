"""B4 tests execute the actual loader branch, comparing fixed and reviewed code."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import joblib
import torch
from softsonic_cpu_harness import BASELINE, close, functions, source

PATH = 'gear_sonic/utils/motion_lib/motion_lib_base.py'
resample = functions(PATH, ['_resample_softsonic_metadata'])._resample_softsonic_metadata


def loader(raw, fps, target_fps, revision=None):
    n = len(raw) if fps == target_fps else len(torch.arange(0, (len(raw)-1)*1/fps, 1/target_fps))
    tree = ast.parse(source(PATH, revision))
    branches = [x for x in ast.walk(tree) if isinstance(x, ast.If)
                and ast.unparse(x.test) == 'self.has_softsonic'
                and 'raw_ss =' in ast.unparse(x)]
    assert len(branches) == 1
    motion = NS(global_rotation=torch.zeros(n, 1, 4))
    ns = dict(torch=torch, self=NS(has_softsonic=True, target_fps=target_fps),
              curr_file={'softsonic': raw, 'fps': fps}, SOFTSONIC_FIELD='softsonic',
              to_torch=torch.as_tensor, start=0, end=len(raw), curr_motion=motion,
              _resample_softsonic_metadata=resample)
    exec(compile(ast.Module(body=branches, type_ignores=[]), PATH, 'exec'), ns)
    return motion.softsonic


raw = torch.arange(4.)[:, None]
old, new = loader(raw, 30, 50, BASELINE), loader(raw, 30, 50)
close(old[:, 0], [0., 1., 2., 2., 3.])
close(new[:, 0], [0., 1., 1., 2., 2.])
print('B4 source indices at 0/20/40/60/80ms:', old.flatten().tolist(), '->', new.flatten().tolist())
# Same FPS is exact; downsampling and short clips use the same FK timestamps.
for count, fps, target in [(4, 30, 30), (11, 50, 30), (3, 30, 40), (2, 30, 50)]:
    data = torch.arange(float(count))[:, None]
    result = loader(data, fps, target)
    times = torch.arange(count) / fps if fps == target else torch.arange(0, (count-1)*1/fps, 1/target)
    close(result[:, 0], (times*fps).round())
try:
    resample(raw, 30, 50, 6)
except ValueError:
    pass
else:
    raise AssertionError('FK time/count mismatch must fail')

ap = argparse.ArgumentParser()
ap.add_argument('--motion-dir', type=Path)
args = ap.parse_args()
if args.motion_dir:
    frames = mismatches = activity = body = 0
    max_force_delta = max_time_shift = 0.
    for path in sorted(args.motion_dir.glob('*.pkl')):
        for record in joblib.load(path).values():
            data = torch.as_tensor(record['softsonic'])
            old, new = loader(data, record['fps'], 50, BASELINE), loader(data, record['fps'], 50)
            ids = torch.arange(float(len(data)))[:, None]
            oi, ni = loader(ids, record['fps'], 50, BASELINE), loader(ids, record['fps'], 50)
            frames += len(new)
            mismatches += int((oi != ni).sum())
            activity += int(((old[:, 7] > 0) != (new[:, 7] > 0)).sum())
            body += int(((old[:, 6] != new[:, 6]) | ((old[:, 7] > 0) != (new[:, 7] > 0))).sum())
            max_force_delta = max(max_force_delta, float((old[:, :3]-new[:, :3]).norm(dim=1).max()))
            n = len(new)
            max_time_shift = max(max_time_shift, float((torch.linspace(0, (len(data)-1)/record['fps'], n) - torch.arange(n)/50).abs().max()))
    print(dict(frames=frames, changed_indices=mismatches, activity_changes=activity,
               activity_or_body_changes=body, max_force_delta_N=max_force_delta,
               max_old_grid_shift_ms=1000*max_time_shift))
print('B4 PASS')
