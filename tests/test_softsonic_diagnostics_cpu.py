"""Execute production AST blocks on CPU; no Isaac simulator imports required."""
import ast
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import textwrap
import unittest
from unittest.mock import patch
import torch

ROOT = Path(os.environ.get("SOFTSONIC_TEST_REPO", Path(__file__).resolve().parents[1]))
WRAPPER = (ROOT/'gear_sonic/envs/wrapper/manager_env_wrapper.py').read_text()

def run(source, scope):
    exec(compile(textwrap.dedent(source), 'production_block', 'exec'), scope)

class Diagnostics(unittest.TestCase):
    def test_velocity_for_plain_aug_and_body_dof_mapping(self):
        text=(ROOT/'gear_sonic/envs/manager_env/mdp/commands.py').read_text()
        code=text[text.index('        # Compare only body joints'):text.index('    def resample_all_commands')]
        for mismatch in [False,True]:
            for aug in [False,True]:
                with self.subTest(mismatch=mismatch,aug=aug):
                    robot=torch.tensor([[10.,2.,20.,4.]]) if mismatch else torch.tensor([[2.,4.]])
                    obj=NS(has_dof_mismatch=mismatch,body_joint_indices=[1,3],robot_joint_pos=robot,robot_joint_vel=robot,
                        joint_pos=torch.ones(1,2),joint_vel=torch.ones(1,2),metrics={},
                        motion_lib=NS(has_aug_pose=aug,get_dof_pos_aug=lambda *a:torch.zeros(1,2)),
                        motion_start_time_steps=torch.zeros(1),time_steps=torch.zeros(1),motion_ids=torch.zeros(1),
                        body_pos_w_aug=torch.zeros(1,2,3),robot_body_pos_w=torch.zeros(1,2,3))
                    run(code,{'self':obj,'torch':torch})
                    self.assertEqual(obj.metrics['error_joint_vel'].item(),2.)

    def test_dump_once_accumulated_and_failed_write(self):
        start=WRAPPER.index('        dump_path = os.environ.get("SOFTSONIC_DUMP_ATM_OBS")')
        code='def dump(self, atm_obs_dict):\n'+WRAPPER[start:WRAPPER.index('    def reset_all',start)]
        ns={'torch':torch,'os':os,'logger':NS(info=lambda *a:None,warning=lambda *a:None)}
        exec(compile(code,'production_dump','exec'),ns)
        for every,count,expected in [(0,1,[2.]),(2,3,[2.,4.,6.])]:
            obj=NS(env=NS())  # optional force/target fields deliberately unavailable
            saves=[]
            with patch.dict(os.environ,{'SOFTSONIC_DUMP_ATM_OBS':'unused','SOFTSONIC_DUMP_ATM_OBS_AFTER':'2',
                    'SOFTSONIC_DUMP_ATM_OBS_EVERY':str(every),'SOFTSONIC_DUMP_ATM_OBS_COUNT':str(count)}), \
                    patch.object(torch,'save',lambda payload,path:saves.append(payload)):
                for step in range(1,11): ns['dump'](obj,{'token':torch.tensor([[float(step)]])})
            self.assertEqual(len(saves),1)
            self.assertEqual(saves[0]['token'].flatten().tolist(),expected)
        # A failed write after collecting COUNT must retry the original batch.
        obj=NS(env=NS()); attempts=[]
        def transient_save(payload, path):
            attempts.append(payload['token'].flatten().tolist())
            if len(attempts)==1:
                raise OSError('transient write failure')
        with patch.dict(os.environ,{'SOFTSONIC_DUMP_ATM_OBS':'unused','SOFTSONIC_DUMP_ATM_OBS_AFTER':'2',
                'SOFTSONIC_DUMP_ATM_OBS_EVERY':'2','SOFTSONIC_DUMP_ATM_OBS_COUNT':'3'}), \
                patch.object(torch,'save',transient_save):
            for step in range(1,11):
                try:
                    ns['dump'](obj,{'token':torch.tensor([[float(step)]])})
                except OSError:
                    self.assertEqual(step,6)
        self.assertEqual(attempts,[[2.,4.,6.],[2.,4.,6.]])
        self.assertEqual(len(obj._softsonic_dump_buf),3)
        for every in [0,1]:
            obj=NS(env=NS())
            with patch.dict(os.environ,{'SOFTSONIC_DUMP_ATM_OBS':'unused','SOFTSONIC_DUMP_ATM_OBS_AFTER':'1',
                    'SOFTSONIC_DUMP_ATM_OBS_EVERY':str(every),'SOFTSONIC_DUMP_ATM_OBS_COUNT':'1'}), \
                    patch.object(torch,'save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError): ns['dump'](obj,{'token':torch.ones(1,1)})
                self.assertFalse(getattr(obj,'_softsonic_atm_obs_dumped',False))

    def test_plain_reload_getters_cannot_read_previous_aug_batch(self):
        import joblib
        import numpy as np
        src=(ROOT/'gear_sonic/utils/motion_lib/motion_lib_base.py').read_text()
        tree=ast.parse(src)
        original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MotionLibBase')
        pairs=[('dof_pos_aug','dof_pos'),('dof_vel_aug','dof_vel'),
               ('body_lin_vel_w_aug','body_lin_vel_w'),('body_ang_vel_w_aug','body_ang_vel_w')]
        names={'get_'+name for pair in pairs for name in pair}
        methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in names]
        cls=ast.ClassDef(name='MotionAccessors',bases=[],keywords=[],body=methods,decorator_list=[])
        namespace={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'production_getters','exec'),namespace)
        obj=namespace['MotionAccessors']();obj.m_cfg={};obj.length_starts=torch.tensor([0,2])
        obj.has_aug_pose=True;obj.has_softsonic=True
        for aug,ref in pairs:
            setattr(obj,aug,torch.full((5,2),777.))
            setattr(obj,ref,torch.arange(10.).reshape(5,2))
        contract={}
        exec(compile((ROOT/'gear_sonic/utils/motion_lib/softsonic_contract.py').read_text(),'production_contract','exec'),contract)
        assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and 'inspect_motion_batch(' in ast.unparse(n)]
        self.assertEqual(len(assignments),1)
        parent=compile(ast.Module(body=assignments,type_ignores=[]),'production_capability_assignment','exec')
        plain=dict(pose_aa=np.zeros((3,30,3)),root_trans_offset=np.zeros((3,3)),fps=30.)
        augmented=dict(**plain,pose_aa_aug=plain['pose_aa'].copy(),root_trans_aug=plain['root_trans_offset'].copy(),softsonic=np.zeros((3,15)))
        ids=torch.tensor([1]);steps=torch.tensor([1])
        scope=dict(self=obj,inspect_motion_batch=contract['inspect_motion_batch'],joblib=joblib,is_evaluation=False)
        for record,want_aug in [(augmented,True),(plain,False),(augmented,True)]:
            scope['motion_data_list']=[record]
            exec(parent,scope)
            self.assertEqual(obj.has_aug_pose,want_aug)
            for aug,ref in pairs:
                actual=getattr(obj,'get_'+aug)(ids,steps)
                expected=getattr(obj,aug if want_aug else ref)[[3]]
                torch.testing.assert_close(actual,expected)

    def test_progress_thresholds_empty_sets_and_historical_mean(self):
        src=(ROOT/'gear_sonic/envs/manager_env/mdp/rewards.py').read_text()
        fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='softsonic_compliance_progress')
        scope={'torch':torch}; exec(compile(ast.Module(body=[fn],type_ignores=[]),'production_progress','exec'),scope)
        function=scope['softsonic_compliance_progress']
        math=ModuleType('isaaclab.utils.math'); math.quat_rotate=lambda q,v:v.clone()
        demand=torch.tensor([.01,.010001,.04999,.05,.050001,.2],dtype=torch.float64)
        progress=torch.tensor([0.,-.5,.2,.5,.6,.8],dtype=torch.float64)
        pos=torch.zeros(6,1,3,dtype=torch.float64);pos[:,0,0]=demand
        robot=pos.clone();robot[:,0,0]=demand*progress
        ml=NS(get_time_step_total=lambda ids:torch.ones(6,dtype=torch.long),get_body_pos_w_aug_full=lambda *a:pos,
            get_body_pos_w_full=lambda *a:torch.zeros_like(pos))
        cmd=NS(motion_lib=ml,motion_ids=torch.arange(6),motion_start_time_steps=torch.zeros(6,dtype=torch.long),time_steps=torch.zeros(6,dtype=torch.long))
        class Scene(dict): pass
        scene=Scene(robot=NS(data=NS(body_pos_w=robot)));scene.env_origins=torch.zeros(6,3,dtype=torch.float64)
        env=NS(_softsonic_active_force_body=torch.zeros(6,dtype=torch.long),scene=scene,
            command_manager=NS(get_term=lambda name:cmd),_softsonic_ff_anchor_rot=torch.zeros(6,4),
            _softsonic_ff_anchor_pos=torch.zeros(6,3,dtype=torch.float64),_softsonic_ff_anchor_ref_pos=torch.zeros(6,3,dtype=torch.float64))
        start=WRAPPER.index('                    _prog, _den = softsonic_compliance_progress')
        code=WRAPPER[start:WRAPPER.index('                except Exception as _exc',start)]
        with patch.dict('sys.modules',{'isaaclab.utils.math':math}):
            actual,den=function(env);torch.testing.assert_close(actual,progress[1:]);torch.testing.assert_close(den,demand[1:])
            for ids,want1,want5 in [(list(range(6)),5,2),([],0,0),([1],1,0)]:
                env._softsonic_active_force_body.fill_(-1);env._softsonic_active_force_body[ids]=0
                logs={'softsonic/compliance_progress_count':torch.tensor(0.),'softsonic/compliance_progress_gt5cm_count':torch.tensor(0.)}
                run(code,{'self':NS(env=env),'extras':{'to_log':logs},'softsonic_compliance_progress':function})
                self.assertEqual(logs['softsonic/compliance_progress_count'].item(),want1)
                self.assertEqual(logs['softsonic/compliance_progress_gt5cm_count'].item(),want5)
                if want5:
                    self.assertAlmostEqual(logs['softsonic/compliance_progress'].item(),.32)
                    self.assertAlmostEqual(logs['softsonic/compliance_progress_gt5cm'].item(),.7)
                else:self.assertNotIn('softsonic/compliance_progress_gt5cm',logs)
                if not want1:self.assertNotIn('softsonic/compliance_progress',logs)

if __name__=='__main__': unittest.main()
