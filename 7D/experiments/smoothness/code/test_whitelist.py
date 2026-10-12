"""Ensure every penalty evaluation stays on the whitelist and gradients remain valid."""
from pathlib import Path
import importlib.util,json,sys,types
import torch
from whitelist_pairs import ALLOWED_PAIRS
import train_sixd_single_longrun as new
from common import ModelConfig,build_model,constant_baseline_rule
norm=json.loads(Path(sys.argv[1]).read_text());root=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
cfg=types.SimpleNamespace(feature_mode='tf_sf7d',target='S',tfsf_unimodal_max_points=512,tfsf_unimodal_prominence_fraction=.02)
def normalized(pairs):
 x=torch.zeros(len(pairs),7)
 for i,(tf,sf) in enumerate(pairs):
  x[i,5]=new.normalized_feature_value(norm,'lgnTF',tf);x[i,6]=new.normalized_feature_value(norm,'lgnSF',sf)
 return x
x=normalized(ALLOWED_PAIRS);seen=[]
class Spy(torch.nn.Module):
 def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(.2))
 def forward(self,z):
  raw=torch.stack([z[:,5]*norm['lgnTF']['std']+norm['lgnTF']['mean'],z[:,6]*norm['lgnSF']['std']+norm['lgnSF']['mean']],dim=1)
  allowed=raw.new_tensor(ALLOWED_PAIRS);assert torch.isclose(raw[:,None,:],allowed[None,:,:],atol=1e-5,rtol=0).all(2).any(1).all()
  seen.extend(raw.detach().tolist());return torch.nn.functional.softplus(self.w*(z[:,5]**2+z[:,6]))
m=Spy()
for t in 'SCI':
 cfg.target=t
 a,b=new.tfsf_unimodality_losses(m,x,cfg,norm);assert torch.isfinite(a+b)
 (a+b).backward();assert torch.isfinite(m.w.grad);m.zero_grad()
 with torch.no_grad():assert all(torch.isfinite(v) for v in new.tfsf_unimodality_losses(m,x,cfg,norm))
cfg.target='S'
# Two SF points at TF2: no invented interior peak and no out-of-list evaluation.
a,b=new.tfsf_unimodality_losses(m,normalized([(2,2),(2,2.5)]),cfg,norm);assert b.item()==0
try:new.tfsf_unimodality_losses(m,normalized([(32,2)]),cfg,norm)
except ValueError:pass
else:raise AssertionError('Forbidden anchor accepted')
# Same penalty algebra as parent when allowed grids coincide with its grids.
spec=importlib.util.spec_from_file_location('parent_trainer',root/'source_trainer.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
import whitelist_pairs
saved=whitelist_pairs.ALLOWED_PAIRS;full=tuple((tf,sf) for tf in [.5,1,2,4,10,16,24] for sf in [.5,1,2,2.5,3,5,10]);whitelist_pairs.ALLOWED_PAIRS=full
plain=torch.nn.Sequential(torch.nn.Linear(7,8),torch.nn.Tanh(),torch.nn.Linear(8,1),torch.nn.Softplus())
z=normalized(full);before=old.tfsf_unimodality_losses(plain,z,cfg,norm);after=new.tfsf_unimodality_losses(plain,z,cfg,norm)
for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-8)
whitelist_pairs.ALLOWED_PAIRS=saved
# Exact baseline inference behavior retained, independently of the loss whitelist.
for t,rate in [('S',2.5),('C',4.5),('I',12.5)]:
 c=ModelConfig(target=t,feature_mode='tf_sf7d',input_dim=7,hidden_dims=(128,)*4,activation_seq=('tanh',)*4);c.constant_baseline_rule=constant_baseline_rule(norm,t);model=build_model(c)
 raw=torch.tensor([[1,0,20,20000,10000,10,2.5],[1,50,20,20000,10000,0,2.5],[1,50,20,20000,10000,10,0],[1,50,20,20000,10000,51,2.5],[1,50,20,20000,10000,10,16]],dtype=torch.float32)
 z=raw.clone()
 for j,k in enumerate(['OriCos2','Contrast','L6','L4E','L4I','lgnTF','lgnSF']):
  if j:z[:,j]=(raw[:,j]-norm[k]['mean'])/norm[k]['std']
 torch.testing.assert_close(model(z),torch.full((5,),rate),rtol=0,atol=0)
(root/'loss_validation.json').write_text(json.dumps({'passed':True,'penalty_forward_pairs_verified':len(seen),'all_three_targets_backward_finite':True,'validation_no_grad_passed':True,'two_point_SF_penalty_zero':True,'forbidden_anchor_rejected':True,'parent_formula_regression_passed':True,'five_baseline_triggers_exact_all_targets':True},indent=2))
print('Whitelist loss/gradient/formula/baseline checks passed')
