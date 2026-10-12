"""Store native direct-LIF rows in one resumable HDF5 bundle per source unit."""
from pathlib import Path
import hashlib,json,sys
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def read_block(path):
 with h5py.File(path) as f:
  d={k:f[k][:].ravel() for k in ['directL4E','directL4I','directS','directC','directI','sample_inds']}
  d['I_angles']=f['I_angles'][:].ravel()
  for k in ['Angle','Contr','lgnTF','lgnSF','DownFactor','LGNctgr','L6ctgr','S_angle']:d[k]=float(f[k][:].ravel()[0])
 return d

def check_block(d,u,l6,grid):
 for k in ['directL4E','directL4I','directS','directC','directI','sample_inds']:
  assert len(d[k])==615 and np.isfinite(d[k]).all(),k
 assert np.array_equal(d['sample_inds'].astype(int),grid['sample_inds'])
 assert np.array_equal(d['directL4E'],grid['directL4E']) and np.allclose(d['directL4I'],grid['directL4I'],rtol=0,atol=1e-8)
 assert (d['directL4E']>=0).all() and (d['directL4I']>=0).all()
 for k,v in [('Angle',u['angle']),('Contr',u['contrast']),('lgnTF',u['tf']),('lgnSF',u['sf']),('DownFactor',10),('LGNctgr',5 if u['background'] else 1),('L6ctgr',l6)]:assert d[k]==v,(k,d[k],v)
 assert np.array_equal(d['I_angles'],np.arange(90,44,-7.5))
 if not u['background']:assert d['S_angle']==u['gamma']
 return True

def pack(unit,l6,path,reuse):
 u=json.loads((ROOT/'source_units.json').read_text())[unit];grid=json.loads((ROOT/'grid_validation.json').read_text())
 d=read_block(path);check_block(d,u,l6,grid)
 reused=0;reuse_sha=''
 if reuse:
  with h5py.File(reuse) as f:old={k:f[k][:].ravel() for k in ['sample_inds','directL4E','directL4I','directS','directC','directI']}
  loc=np.searchsorted(d['sample_inds'],old['sample_inds']);assert len(loc)==128
  for k in ['sample_inds','directL4E','directL4I','directS','directC','directI']:assert np.array_equal(d[k][loc],old[k]),('retained128 changed',k)
  reused=len(loc);reuse_sha=sha(reuse)
 bundle=ROOT/'source_bundles'/f'unit_{unit:04d}.h5'
 with h5py.File(bundle,'a') as f:
  key=f'L6_{l6:02d}';assert key not in f,'Concurrent or duplicate source block'
  if '_writing' in f:del f['_writing']
  g=f.create_group('_writing')
  for k,v in d.items():g.create_dataset(k,data=v,compression='gzip' if np.ndim(v) else None)
  g.attrs.update(raw_mat_sha256=sha(path),reused_points=reused,reuse_source=str(reuse),reuse_sha256=reuse_sha,verified=True)
  f.flush();f.move('_writing',key);f.flush()
 print('PACKED',unit,l6,615,'reused',reused,flush=True)

if __name__=='__main__':pack(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4] if len(sys.argv)>4 else '')
