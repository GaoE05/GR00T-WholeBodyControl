"""Analytic unit-quaternion oracle and compatibility for both SLERP APIs."""
import ast
import math
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gear_sonic.isaac_utils.rotations import slerp as isaac_slerp
from gear_sonic.trl.utils.math import slerp as trl_slerp

def zquat(angle,dtype):
    angles=torch.as_tensor(angle,dtype=dtype);q=torch.zeros((*angles.shape,4),dtype=dtype)
    q[...,0]=torch.cos(angles/2);q[...,3]=torch.sin(angles/2)
    return q

def legacy(path):
    source=subprocess.check_output(['git','show','6752286:'+path],cwd=ROOT,text=True)
    tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='slerp');node.decorator_list=[]
    ns={'torch':torch,'Tensor':torch.Tensor};exec(compile(ast.Module(body=[node],type_ignores=[]),'legacy-slerp','exec'),ns)
    return ns['slerp']

class Slerp(unittest.TestCase):
    def check_oracle(self,angle,dtype,sign=1):
        times=torch.tensor([0.,.2,.8,1.],dtype=dtype);q0=zquat(torch.zeros_like(times),dtype);q1=zquat(torch.full_like(times,angle),dtype)*sign
        expected=zquat(times*angle,dtype)
        for name,function,t in [('isaac',isaac_slerp,times[:,None]),('trl',trl_slerp,times)]:
            actual=function(q0,q1,t)
            self.assertTrue(torch.isfinite(actual).all(),name)
            tolerance=2e-7 if dtype==torch.float32 else 1e-10
            # Normal-angle arithmetic is intentionally bitwise legacy-compatible;
            # compare rotations after normalization and separately bound norm drift.
            unit=actual/actual.norm(dim=-1,keepdim=True)
            torch.testing.assert_close(unit,expected,rtol=0,atol=tolerance)
            norm_tolerance=2e-6 if dtype==torch.float32 and angle>.0015 else tolerance
            torch.testing.assert_close(actual.norm(dim=-1),torch.ones_like(times),rtol=0,atol=norm_tolerance)
            torch.testing.assert_close(unit[0],q0[0],rtol=0,atol=tolerance)
            torch.testing.assert_close(unit[-1],q1[-1]*sign,rtol=0,atol=tolerance)
    def test_known_small_angle_endpoints_nonmidpoints_and_negative_sign(self):
        for dtype in (torch.float32,torch.float64):
            for angle in (0.,1e-7,.0001,.0015):
                for sign in (1,-1):self.check_oracle(angle,dtype,sign)
    def test_small_to_normal_boundary_and_large_rotations(self):
        boundary=2*math.asin(.001)
        for dtype in (torch.float32,torch.float64):
            for angle in (boundary*(1-1e-5),boundary,boundary*(1+1e-5),.2,1.7,math.pi-.001):
                self.check_oracle(angle,dtype)
    def test_scalar_and_higher_rank_broadcast_interfaces(self):
        for dtype in (torch.float32,torch.float64):
            q0=zquat(torch.zeros(2,3),dtype);q1=zquat(torch.full((2,3),.0015),dtype)
            for t in (torch.tensor(.2,dtype=dtype),torch.tensor([.2,.8],dtype=dtype)[:,None,None]):
                got=isaac_slerp(q0,q1,t);expected=zquat(torch.zeros((2,3),dtype=dtype)+t.squeeze(-1)*.0015 if t.ndim else torch.full((2,3),float(t)*.0015,dtype=dtype),dtype)
                torch.testing.assert_close(got,expected,rtol=0,atol=2e-7 if dtype==torch.float32 else 1e-10)
            q0=q0[0];q1=q1[0];t=torch.tensor([0.,.2,1.],dtype=dtype)
            self.assertEqual(trl_slerp(q0,q1,t).shape,(3,4))
    def test_general_axis_composition_matches_independent_rotation_oracle(self):
        base=Rotation.from_rotvec([.3,-.4,.2]);axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
        for dtype in (torch.float32,torch.float64):
            for angle in (.0015,.3,2.7):
                times=np.array([0.,.2,.8,1.]);end=base*Rotation.from_rotvec(axis*angle)
                q0=torch.tensor(np.tile(base.as_quat()[[3,0,1,2]],(4,1)),dtype=dtype);q1=torch.tensor(np.tile(end.as_quat()[[3,0,1,2]],(4,1)),dtype=dtype)
                expected=base*Rotation.from_rotvec(times[:,None]*axis[None]*angle)
                for fn,t in ((isaac_slerp,torch.tensor(times,dtype=dtype)[:,None]),(trl_slerp,torch.tensor(times,dtype=dtype))):
                    actual=fn(q0,q1,t).double().numpy()[:,[1,2,3,0]]
                    error=(Rotation.from_quat(actual)*expected.inv()).magnitude()
                    self.assertLess(error.max(),5e-7 if dtype==torch.float32 else 3e-10)
    def test_normal_branch_is_bitwise_compatible_with_original(self):
        torch.manual_seed(7)
        for dtype in (torch.float32,torch.float64):
            a=torch.randn(100,4,dtype=dtype);a/=a.norm(dim=-1,keepdim=True)
            b=torch.randn(100,4,dtype=dtype);b/=b.norm(dim=-1,keepdim=True)
            t=torch.rand(100,dtype=dtype)
            self.assertTrue(((a*b).sum(-1).abs()<.999).all())
            for path,fn,time in [('gear_sonic/isaac_utils/rotations.py',isaac_slerp,t[:,None]),('gear_sonic/trl/utils/math.py',trl_slerp,t)]:
                self.assertTrue(torch.equal(fn(a,b,time),legacy(path)(a,b,time)),path)
    def test_roundoff_above_one_is_finite_and_retains_interpolation(self):
        for dtype in (torch.float32,torch.float64):
            q0=zquat([0.],dtype)*(1+4*torch.finfo(dtype).eps);q1=zquat([1e-7],dtype)*(1+4*torch.finfo(dtype).eps)
            for fn,t in ((isaac_slerp,torch.tensor(.8,dtype=dtype)),(trl_slerp,torch.tensor([.8],dtype=dtype))):
                out=fn(q0,q1,t);self.assertTrue(torch.isfinite(out).all());self.assertGreater(float(out[0,3]),0.)
                torch.testing.assert_close(out,zquat([.8e-7],dtype),rtol=0,atol=2e-7 if dtype==torch.float32 else 1e-12)
    def test_coincident_and_small_angle_gradients_remain_finite(self):
        for dtype in (torch.float32,torch.float64):
            for angle in (0.,1e-7,.0015):
                for fn,t in ((isaac_slerp,torch.tensor([[.2]],dtype=dtype)),(trl_slerp,torch.tensor([.2],dtype=dtype))):
                    a=zquat([0.],dtype).requires_grad_();b=zquat([angle],dtype).requires_grad_();t=t.clone().requires_grad_()
                    fn(a,b,t).sum().backward()
                    self.assertTrue(all(torch.isfinite(x.grad).all() for x in (a,b,t)))

if __name__=='__main__':unittest.main()
