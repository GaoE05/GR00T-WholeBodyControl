"""B9/N2 rejection and reload tests, no simulator or CUDA required."""
import ast
from copy import deepcopy
from pathlib import Path
from queue import Queue
import sys
from types import SimpleNamespace

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gear_sonic.utils.motion_lib.softsonic_contract import inspect_motion_batch, get_worker_result


def inspect(records, cfg=None, evaluation=False):
    return inspect_motion_batch(records, cfg or {}, evaluation, joblib.load)


def rejects(records, text, cfg=None):
    try:
        inspect(records, cfg)
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"Expected rejection: {text}")


plain = dict(pose_aa=np.zeros((3, 30, 3)), root_trans_offset=np.zeros((3, 3)), fps=30.)
aug = dict(**plain, pose_aa_aug=plain['pose_aa'].copy(),
           root_trans_aug=plain['root_trans_offset'].copy(), softsonic=np.zeros((3, 15)))
assert inspect([plain]) == (False, False)
assert inspect([aug, aug]) == (True, True)
rejects([aug], "zero_root_xy", {"zero_root_xy": True})
try:
    inspect([aug], {"zero_root_xy": True}, True)
except ValueError as error:
    assert "zero_root_xy" in str(error)
else:
    raise AssertionError('Evaluation must also reject zero_root_xy')
assert inspect([plain], {"zero_root_xy": True}, True) == (False, False)
for records in ([plain, aug], [aug, plain]):
    rejects(records, "mixed SoftSONIC schema")
for name in ("freeze_frame_aug", "randomize_heading", "randomize_wrist_poses", "cat_upper_body_poses"):
    rejects([aug], "not synchronized", {name: True})
    assert inspect([plain], {name: True}) == (False, False)
    assert inspect([aug], {name: True}, True) == (True, True)
for key, value, text in (("root_trans_aug", np.zeros((2,3)), "shape"),
                         ("softsonic", np.zeros((3,14)), "shape"),
                         ("fps", 0, "fps")):
    bad=deepcopy(aug); bad[key]=value; rejects([bad], text)
bad=deepcopy(aug); del bad['root_trans_aug']; rejects([bad], "together")
bad=deepcopy(aug); bad['softsonic'][0,0]=np.nan; rejects([bad], "non-finite")
bad=deepcopy(aug); bad['softsonic'][0,7]=-1; rejects([bad], "negative")
for index in (-1, .5, 5):
    bad=deepcopy(aug); bad['softsonic'][0,8]=1; bad['softsonic'][0,6]=index
    rejects([bad], "body id")
# Execute the parent assignment from the production loader, including reloading plain data.
source=Path(__file__).resolve().parents[1]/'utils/motion_lib/motion_lib_base.py'
tree=ast.parse(source.read_text())
nodes=[n for n in ast.walk(tree) if isinstance(n, ast.Assign)
       and 'inspect_motion_batch(' in ast.unparse(n)]
assert len(nodes)==1
state=SimpleNamespace(has_aug_pose=True, has_softsonic=True, m_cfg={})
scope=dict(self=state, inspect_motion_batch=inspect_motion_batch, joblib=joblib, is_evaluation=False)
for records, expected in (([aug], (True,True)), ([plain], (False,False)), ([aug], (True,True))):
    scope['motion_data_list']=records
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    assert (state.has_aug_pose,state.has_softsonic)==expected
q=Queue(); q.put({'ok':1}); assert get_worker_result(q, [])=={'ok':1}
for exitcode in (1, -9, 0):
    try:
        get_worker_result(Queue(), [SimpleNamespace(pid=12, exitcode=exitcode)])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Dead worker must propagate an error')
if len(sys.argv)>1:
    records=[{'path':str(p)} for p in sorted(Path(sys.argv[1]).glob('*.pkl'))]
    assert inspect(records)==(True,True)
    # A production-sized repeated batch is validated in the parent, independently of order.
    assert inspect(records*14)==(True,True)
    print(f'Validated {len(records)} actual source files and a 140-entry batch')
print('B9/N2 PASS: schema order/reload, finite/shape/id/fps, config gates, dead workers')
