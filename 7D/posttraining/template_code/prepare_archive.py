from pathlib import Path
import json,csv,hashlib,shutil
import numpy as np
from scipy.io import loadmat,savemat
P=Path(__file__).resolve().parent.parent;out=P/'local_archive';out.mkdir(exist_ok=True);hc=out/'HC norm';hc.mkdir(exist_ok=True)
rt=P/'atlas_139/runtime_h96iter';meta=json.loads((P/'checkpoint_provenance.json').read_text());prov=json.loads((P/'atlas_139/runtime_provenance.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for t in 'SCI':assert meta[t]['sha256']==prov['models'][t]['sha256']==sha(P/'checkpoints'/(t+'.pt'))
for f in rt.glob('*.m'):
 if f.name.startswith(('predict_','LocalResponse_')):shutil.copy2(f,out/f.name)
plots=P/'dense_tuning/plots';jobs=[{'out':str(out),'orientation':str(P/'tuning_curves/orientation_tuning_contr95_default3pixels.csv'),'contrast0':str(plots/'contrast_tuning_angle0_pixel5_10.csv'),'contrast22':str(plots/'contrast_tuning_angle22p5_pixel1_10.csv'),'tf0':str(plots/'tf_tuning_angle0_contr100_pixel5_10.csv'),'tf22':str(plots/'tf_tuning_angle0_contr100_pixel1_10.csv'),'sf0':str(plots/'sf_tuning_angle0_contr100_pixel5_10.csv'),'sf22':str(plots/'sf_tuning_angle0_contr100_pixel1_10.csv'),'parity':str(P/'parity_inputs.mat'),'fixedbaseline':True,'suffix':''}]
for k,v in jobs[0].items():
 if k not in ['out','fixedbaseline','suffix']:assert Path(v).is_file(),v
(P/'plot_jobs.json').write_text(json.dumps(jobs,indent=2))
def same(a,b):
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
  a=np.asarray(a);b=np.asarray(b);return a.shape==b.shape and all(same(x,y) for x,y in zip(a.flat,b.flat))
 if isinstance(a,str) or isinstance(b,str):return a==b
 return bool(np.isclose(a,b,rtol=0,atol=1e-12))
rows=[];jobs=[];stage=P/'comparison_data';stage.mkdir(exist_ok=True)
for ref in json.loads((P/'representative_reference_manifest.json').read_text()):
 i=ref['case'];lp=Path(ref['reference']);mp=Path(ref['maps']);assert sha(lp)==ref['reference_sha256'] and sha(mp)==ref['maps_sha256']
 npath=P/'representative192/data'/f'case_{i:03d}.mat';lib=loadmat(lp,simplify_cells=True);nn=loadmat(npath,simplify_cells=True);maps=loadmat(mp,simplify_cells=True)
 q=np.array([ref[k] for k in ['orientation','contrast','tf','sf']]);assert np.array_equal(q,maps['repQ']);assert np.array_equal([nn['angle'],nn['Contr'],nn['lgnTF'],nn['lgnSF']],q)
 for k in ['p','EKpUse','IKpUse']:assert same(nn[k],lib[k]),(i,k)
 assert same(nn['L6ParamUse'],lib['L6pars'][lib['L6parId']-1]),i
 L=np.stack([lib[k+'_map_fulllib'] for k in 'SCI'],axis=2);N=np.stack([nn[k+'_map_full'] for k in 'SCI'],axis=2);assert np.array_equal(L,maps['repMaps']);assert np.isfinite(L).all() and np.isfinite(N).all()
 L=np.concatenate([L,(.6923*L[:,:,0]+.3077*L[:,:,1])[:,:,None]],axis=2);N=np.concatenate([N,(.6923*N[:,:,0]+.3077*N[:,:,1])[:,:,None]],axis=2)
 err=float(np.sqrt(np.mean(.8*(L[:,:,3]-N[:,:,3])**2+.2*(L[:,:,2]-N[:,:,2])**2)))
 stem='orientation(%g)_contrast(%g)_TF(%g)_SF(%g)'%tuple(q);f=stage/(stem+'.mat');savemat(f,dict(library=L,network=N,condition=q,hcnorm=err,library_residual=maps['repResidual'],nn_residual=nn['final_step_max']))
 jobs.append({'data':str(f),'out':str(hc/(stem+'.pdf')),'stem':stem})
 rows.append(dict(case=i,orientation=q[0],contrast=q[1],TF=q[2],SF=q[3],HCnorm=err,library_final_step_HCnorm=float(maps['repResidual']),NN_final_step_max=float(nn['final_step_max']),library_sha256=sha(lp),NN_sha256=sha(npath),**{t+'_checkpoint_sha256':meta[t]['sha256'] for t in 'SCI'}))
with (hc/'comparisons.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(hc/'README.md').write_text('# Representative library comparisons\n\n'+str(len(rows))+' available local reference conditions at submission. Each PDF compares full-library and this frozen scratch network final states with shared S/C/I/E column color limits. HCnorm=sqrt(mean(0.8*(E_library-E_NN)^2+0.2*(I_library-I_NN)^2)), E=0.6923S+0.3077C; distinct from atlas iteration-average HCnorm. Condition, mapping arrays, p, EKp, IKp and L6 parameters were checked. CSV contains reference, network/checkpoint hashes and convergence diagnostics. Finite states do not establish convergence.\n')
(P/'comparison_jobs.json').write_text(json.dumps(jobs,indent=2))
print('Prepared four tuning PDFs and',len(rows),'representative comparisons')
