from pathlib import Path
import hashlib,json,sys,torch,numpy as np
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'code'));torch.set_num_threads(2)
from common import ModelConfig,TrainConfig,build_model,constant_baseline_rule
import train_sixd_single_longrun as base
import train_lowtfsf_grad01 as trainer
s=json.loads((r/'submitted_jobs.json').read_text());norm=json.loads((Path(s['split'])/'feature_norm.json').read_text())
cfg=ModelConfig(target='C',feature_mode='tf_sf7d',input_dim=7,hidden_dims=(1024,)*4,activation_seq=('relu',)*4)
cfg.constant_baseline_rule=constant_baseline_rule(norm,'C')
torch.manual_seed(42);old=build_model(cfg);before=old.net[0].weight.detach().clone()
torch.manual_seed(s['seed']);m=build_model(cfg);assert not torch.equal(before,m.net[0].weight)
assert sum(p.numel() for p in m.parameters())==3158017
def normalize(x):
 z=x.clone()
 for j,k in enumerate(['OriCos2','Contrast','L6','L4E','L4I','lgnTF','lgnSF']):
  if j:z[:,j]=(x[:,j]-norm[k]['mean'])/norm[k]['std']
 return z
raw=torch.tensor([[1,0,20,20000,10000,10,2.5],[1,50,20,20000,10000,0,2.5],[1,50,20,20000,10000,10,0],[1,50,20,20000,10000,51,2.5],[1,50,20,20000,10000,10,16]],dtype=torch.float32)
torch.testing.assert_close(m(normalize(raw)),torch.full((5,),4.5),rtol=0,atol=0)
raw=torch.tensor([[1,50,20,20000,10000,tf,sf] for tf,sf in s['allowed_tf_sf_pairs']],dtype=torch.float32)
lc=TrainConfig(train_mat='',split_dir=s['split'],out_dir='',target='C',device='cpu',feature_mode='tf_sf7d',w_log=.5,w_rel=.5,variance_weight_mode='soft_inv_sem',tf_unimodal_lambda=5,sf_unimodal_lambda=5,seed=s['seed'])
pred,terms=trainer.compute_loss_terms(m,normalize(raw),torch.full((len(raw),),10.),{'FR_ResponseVar':np.zeros(len(raw)),'FR_ResponseN':np.ones(len(raw))},lc,torch.device('cpu'),None,norm)
loss,_,_=base.build_total_loss_from_terms(terms,lc);assert torch.isfinite(loss) and torch.isfinite(pred).all();loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
assert m.net[0].weight.grad.abs().sum()>0
result=dict(passed=True,seed=s['seed'],different_from_seed42=True,parameters=3158017,exact_C_baselines_passed=5,finite_regularized_backward=True,loss=float(loss),first_layer_gradient_norm=float(m.net[0].weight.grad.norm()),scientific_code_unchanged=True)
(r/'seed_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

