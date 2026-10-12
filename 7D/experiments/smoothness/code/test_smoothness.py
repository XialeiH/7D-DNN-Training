"""Analytic, boundary, derivative and parent-objective integration checks."""
import json
import math
import sys
from pathlib import Path
import tempfile
import torch
from smoothness import SETTINGS, AXES, coordinates, to_model, second_difference, curvature_terms, balanced_curvature

torch.set_num_threads(2)
norm = json.loads(Path(sys.argv[1]).read_text())
h = torch.tensor([.01, .025, .05], dtype=torch.float64)
z = torch.tensor([.2, .4, .6], dtype=torch.float64)
assert torch.allclose(second_difference((z-h)**2,z*z,(z+h)**2,h),torch.full_like(h,2.),atol=1e-10)
assert torch.allclose(second_difference(z-h,z,z+h,h),torch.zeros_like(h),atol=1e-10)
xb = torch.zeros(32,7,dtype=torch.float64)
lo,width,mu,sd = coordinates(norm,xb)


class Analytic(torch.nn.Module):
    def __init__(self, quadratic=False, wiggle=False):
        super().__init__(); self.amplitude=torch.nn.Parameter(torch.tensor(.03,dtype=torch.float64))
        self.quadratic=quadratic;self.wiggle=wiggle;self.inputs=[]
    def forward(self,x):
        self.inputs.append((x*sd+mu).detach())
        u=(x*sd+mu-lo)/width
        signal=u[:,5].square() if self.quadratic else (torch.sin(12*math.pi*u[:,5]) if self.wiggle else torch.zeros_like(u[:,5]))
        return torch.expm1((.3+self.amplitude*signal)*math.log1p(30))


constant=curvature_terms(Analytic(),xb,norm,'S',validation=True)
assert max(float(v) for v in constant.values())<1e-20
q=Analytic(quadratic=True);terms=curvature_terms(q,xb,norm,'S',validation=True)
assert abs(float(terms['tf'])-.06**2)<1e-10
assert max(float(terms[k]) for k in ['orientation','contrast','sf'])<1e-18
assert float(curvature_terms(Analytic(wiggle=True),xb,norm,'S',validation=True)['tf'])>float(terms['tf'])
assert torch.equal(terms['tf'],curvature_terms(q,xb,norm,'S',validation=True)['tf'])
terms['tf'].backward();assert abs(float(q.amplitude.grad)-8*.03)<1e-9
for raw in q.inputs:
    assert raw[:,1].min()>=.1-1e-8 and raw[:,1].max()<=100+1e-8
    assert raw[:,5].min()>=.5-1e-8 and raw[:,5].max()<=32+1e-8
    assert raw[:,6].min()>=.5-1e-8 and raw[:,6].max()<=10+1e-8
u=torch.full((2,7),.5,dtype=torch.float64);u[:,0]=torch.tensor([0.,1.],dtype=torch.float64)
assert torch.allclose(to_model(u,lo,width,mu,sd)[0],to_model(u,lo,width,mu,sd)[1],atol=1e-12)

from common import ModelConfig,TrainConfig,build_model,constant_baseline_rule
import train_smooth as wrapper
from whitelist_pairs import ALLOWED_PAIRS
settings={'targets':{t:{'axis_scales':{k:1. for k in AXES}} for t in 'SCI'}}
wrapper.install(settings)
result={}
for target in 'SCI':
    torch.manual_seed(42)
    cfg=ModelConfig(target=target,feature_mode='tf_sf7d',input_dim=7,hidden_dims=(512,)*5,activation_seq=('relu',)*5)
    cfg.constant_baseline_rule=constant_baseline_rule(norm,target);model=build_model(cfg)
    assert sum(p.numel() for p in model.parameters())==1055233
    raw=torch.tensor([[1,0,20,20000,10000,10,2.5],[1,50,20,20000,10000,0,2.5],[1,50,20,20000,10000,10,0],[1,50,20,20000,10000,51,2.5],[1,50,20,20000,10000,10,16]],dtype=torch.float32)
    _,_,m,s=coordinates(norm,raw);x=(raw-m)/s
    assert torch.equal(model(x),torch.full((5,),{'S':2.5,'C':4.5,'I':12.5}[target]))
    physical=torch.tensor([[.8,50,20,20000,10000,tf,sf] for tf,sf in ALLOWED_PAIRS],dtype=torch.float32);x=(physical-m)/s
    with tempfile.TemporaryDirectory() as out:
        train=TrainConfig(train_mat='',split_dir='',out_dir=out,target=target,device='cpu',feature_mode='tf_sf7d',tf_unimodal_lambda=5,sf_unimodal_lambda=5,curvature_lambda=0)
        y=torch.linspace(2,25,len(x));_,parent=wrapper.low.compute_loss_terms(model,x,y,{},train,torch.device('cpu'),None,norm)
        train.curvature_lambda=.01;_,new=wrapper.low.compute_loss_terms(model,x,y,{},train,torch.device('cpu'),None,norm)
        for key in ['rmse_term','relrmse_raw','tf_unimodal_loss','sf_unimodal_loss']:
            assert torch.equal(parent[key],new[key]),key
        fitting,_,_=wrapper.base.build_total_loss_from_terms(parent,train)
        loss,_,_=wrapper.base.build_total_loss_from_terms(new,train)
        assert torch.allclose(loss-fitting,.01*new['curvature_loss'],atol=1e-6)
        loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.net[0].weight.grad.abs().sum()>0
        torch.optim.AdamW(model.parameters(),lr=2e-4).step();assert all(torch.isfinite(p).all() for p in model.parameters())
        result[target]={'finite_original_plus_smooth_backward':True,'baseline_checks':5,'loss':float(loss.detach())}
# The inherited plateau calculation is flat=0, improving>threshold.
assert wrapper.base.stage_improve([1.]*200)==0
assert wrapper.base.stage_improve([1.-i*.001 for i in range(200)])>.001
print(json.dumps({'passed':True,'analytic_curvature_and_gradient':True,'periodic_orientation':True,'fixed_validation_bank':True,'no_baseline_crossing':True,'wiggle_penalized':True,'actual_architecture_checks':result}))
