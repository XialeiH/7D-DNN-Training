"""Use every native source coordinate once, without retained-row subsampling."""
from pathlib import Path
import json,math,shutil,time
import h5py,numpy as np
from source_bundle import sha,check_block
R=Path(__file__).resolve().parents[1];D=R/'data_split_whitelist17';plan=json.loads((R/'plan.json').read_text());units=json.loads((R/'source_units.json').read_text());grid=json.loads((R/'grid_validation.json').read_text())
assert not (D/'dataset_validation.json').exists(),'Dataset already built; reconcile before rerun'
assert len(units)==4267 and len({(u['background'],u['gamma'],u['contrast'],u['tf'],u['sf']) for u in units})==len(units)
assert len(set(zip(grid['directL4E'],grid['directL4I'])))==615
assert len(set(np.cos(np.deg2rad(2*np.array(plan['foreground_gammas']))).astype('f4')))==25
assert len(set(np.cos(np.deg2rad(2*np.array(plan['orientations']))).astype('f4')))==7
fields=['OriCos2','OriSin2','Gamma','Contrast','L6','L4E','L4I','lgnTF','lgnSF','LGNctgr','FR_Response','Source_FR_Response','FR_ResponseVar','FR_ResponseN']
D.mkdir(exist_ok=True);(D/'shards').mkdir(exist_ok=True)
shutil.copy2(Path(plan['reference_split'])/'feature_norm.json',D/'feature_norm.json')
handles={};counts={};source_hashes={};paircounts={};excluded={};source_rows=0
for t in 'SCI':
 counts[t]={s:0 for s in ['train','val','test']};excluded[t]=0;paircounts[t]={f'{a:g},{b:g}':0 for a,b in plan['allowed_tf_sf_pairs']}
 for split in counts[t]:
  p=D/'shards'/f'{t}_{split}.partial.h5';assert not p.exists(),'Partial build exists; must reconcile'
  h=handles[t,split]=h5py.File(p,'w')
  for k in fields:h.create_dataset(k,shape=(0,),maxshape=(None,),dtype='i2' if k=='LGNctgr' else 'f4',chunks=(65536,),compression='gzip',compression_opts=1)
try:
 for unit,u in enumerate(units):
  path=R/'source_bundles'/f'unit_{unit:04d}.h5';source_hashes[str(unit)]=sha(path)
  pieces={(t,split):{k:[] for k in fields} for t in 'SCI' for split in ['train','val','test']}
  with h5py.File(path) as f:
   assert f.attrs.get('complete',False) and f.attrs['status']=='complete'
   for l6 in range(1,41):
    g=f[f'L6_{l6:02d}'];assert g.attrs['verified'];d={k:g[k][()] for k in g};check_block(d,u,l6,grid)
    gammas=plan['orientations'] if u['background'] else [u['gamma']]
    for gi,gamma in enumerate(gammas):
     n=615;source_rows+=n
     meta={k:np.full(n,v,np.float32) for k,v in {'OriCos2':math.cos(math.radians(2*gamma)),'OriSin2':math.sin(math.radians(2*gamma)),'Gamma':gamma,'Contrast':0 if u['background'] else u['contrast'],'L6':l6,'lgnTF':u['tf'],'lgnSF':u['sf'],'FR_ResponseVar':0,'FR_ResponseN':1}.items()}
     meta.update(L4E=d['directL4E'].astype('f4'),L4I=d['directL4I'].astype('f4'),LGNctgr=np.full(n,5 if u['background'] else u['representative_LGNctgr'],np.int16))
     for ti,t in enumerate('SCI'):
      y=d['direct'+t].astype('f4');low,high=plan['target_bounds'][t]
      mask=(meta['L4I']>0)&np.isfinite(y)&(y>=low)&(y<=high);excluded[t]+=int((~mask).sum());idx=np.flatnonzero(mask)
      # Native coordinates/unique source unit/L6/gamma are unique model inputs.
      # Assign each point once to one shard; target bounds differ, split seed remains 42.
      rng=np.random.default_rng(42+unit*1009+l6*1000003+gi*10007+ti*10000019)
      q=rng.random(n);splitids=np.where(q<.8,0,np.where(q<.9,1,2));meta['FR_Response']=y;meta['Source_FR_Response']=y
      paircounts[t][f"{u['tf']:g},{u['sf']:g}"]+=len(idx)
      for si,split in enumerate(['train','val','test']):
       selected=idx[splitids[idx]==si]
       for k in fields:pieces[t,split][k].append(meta[k][selected])
  for (t,split),data in pieces.items():
   arrays={k:np.concatenate(data[k]) for k in fields};start=counts[t][split];end=start+len(arrays['FR_Response']);h=handles[t,split]
   for k in fields:h[k].resize((end,));h[k][start:end]=arrays[k]
   counts[t][split]=end
  if unit%50==0:print('SOURCE_UNITS',unit+1,'/',len(units),counts,flush=True)
finally:
 for h in handles.values():h.close()
# Independently validate all nine on-disk shards in chunks and confirm total accounting.
validation={'verified':False,'source_units':len(units),'native_points_per_block':615,'source_rows_before_target_filter':source_rows,'counts':counts,'excluded':excluded,'pair_counts':paircounts,'normalization_unchanged':sha(D/'feature_norm.json')==sha(Path(plan['reference_split'])/'feature_norm.json'),'split_policy':'disjoint per-input seeded random 80/10/10; all eligible inputs once','no_subsampling':True,'no_interpolation':True,'no_unused_pool_augmentation':True,'source_hashes':source_hashes,'shard_hashes':{}}
for t in 'SCI':
 assert sum(counts[t].values())+excluded[t]==source_rows
 assert all(paircounts[t].values())
 for split in ['train','val','test']:
  path=D/'shards'/f'{t}_{split}.partial.h5'
  with h5py.File(path) as f:
   assert all(len(f[k])==counts[t][split] for k in fields)
   for start in range(0,counts[t][split],200000):
    sl=slice(start,start+200000);x={k:f[k][sl] for k in fields};assert all(np.isfinite(v).all() for v in x.values());assert np.array_equal(x['FR_Response'],x['Source_FR_Response'])
    lo,hi=plan['target_bounds'][t];assert ((x['FR_Response']>=lo)&(x['FR_Response']<=hi)).all();assert (x['L4I']>0).all()
    assert set(zip(x['lgnTF'].tolist(),x['lgnSF'].tolist()))<=set(map(tuple,plan['allowed_tf_sf_pairs']))
  dest=D/'shards'/f'{t}_{split}.h5';path.rename(dest);validation['shard_hashes'][t+'_'+split]=sha(dest)
manifest={'feature_mode':'tf_sf7d','split_mode':'all615_native_unique_input_rows','seed':42,'ratios':{'train':.8,'val':.1,'test':.1},'allowed_tf_sf_pairs':plan['allowed_tf_sf_pairs'],'target_filter':{'inclusive':True,'low_by_target':{t:plan['target_bounds'][t][0] for t in 'SCI'},'high_by_target':{t:plan['target_bounds'][t][1] for t in 'SCI'}},'split_rows_by_target':counts,'shard_paths':{t:{s:str(D/'shards'/f'{t}_{s}.h5') for s in counts[t]} for t in 'SCI'},'native_grid_points':615,'source_units':len(units),'sample_policy':plan['sample_policy']}
(D/'split_manifest.json').write_text(json.dumps(manifest,indent=2));validation['verified']=True;validation['finished_unix']=time.time();(D/'dataset_validation.json').write_text(json.dumps(validation,indent=2));print('DATASET_VERIFIED',counts,flush=True)
