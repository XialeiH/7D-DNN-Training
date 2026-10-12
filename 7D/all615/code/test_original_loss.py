from pathlib import Path
import json,sys,math
import torch
import numpy as np
from common import TrainConfig
import train_sixd_single_longrun as base
root=Path(__file__).resolve().parents[1];norm=json.loads((root/'data_split_whitelist17/feature_norm.json').read_text());torch.set_num_threads(2)
cfg=TrainConfig(train_mat='',split_dir='',out_dir='',target='S',device='cpu',feature_mode='tf_sf7d',w_log=.5,w_rel=.5,raw_mse_lambda=0.,variance_weight_mode='soft_inv_sem')
class FixedPrediction(torch.nn.Module):
 def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor([2.,5.,9.]))
 def forward(self,x):return self.p
m=FixedPrediction();y=torch.tensor([1.,3.,7.]);x=torch.zeros(3,7);x[:,5]=base.normalized_feature_value(norm,'lgnTF',10);x[:,6]=base.normalized_feature_value(norm,'lgnSF',2.5)
meta={'FR_ResponseVar':np.zeros(3),'FR_ResponseN':np.ones(3)}
pred,terms=base.compute_loss_terms(m,x,y,meta,cfg,torch.device('cpu'),None,norm)
total,_,_=base.build_total_loss_from_terms(terms,cfg)
expected=.5*torch.sqrt(((torch.log1p(m.p)-torch.log1p(y))**2).mean()+cfg.loss_eps)+.5*torch.sqrt((((m.p-y)/(y+1))**2).mean()+cfg.loss_eps)
torch.testing.assert_close(total,expected,atol=1e-7,rtol=1e-6)
ga=torch.autograd.grad(total,m.p,retain_graph=True)[0];gb=torch.autograd.grad(expected,m.p)[0];torch.testing.assert_close(ga,gb)
acc=base.WeightedLossAccumulator(cfg);acc.update(y.numpy(),pred.detach().numpy(),meta);v=acc.finalize();assert abs(v['weighted_total_loss']-float(expected))<1e-6
(root/'loss_ratio_validation.json').write_text(json.dumps({'passed':True,'formula':'0.5 weighted log-RMSE + 0.5 weighted relative RMSE','w_log':.5,'w_rel':.5,'raw_mse_lambda':0,'manual_formula_and_gradient_agree':True,'validation_accumulator_agrees':True,'relative_error_denominator':'y+1','test_value':float(total)},indent=2));print('Original log/relative RMSE formula, gradients and validation passed')
