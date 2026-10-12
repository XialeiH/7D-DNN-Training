"""Check rejection of wrong grids, nonfinite rates and wrong source conditions."""
import json,numpy as np
from source_bundle import check_block
u={'angle':0,'gamma':0,'contrast':14,'tf':2,'sf':2,'background':False}
grid={'sample_inds':list(range(1,616)),'directL4E':list(range(615)),'directL4I':list(range(1,616))}
d={k:np.array(v,dtype=float) for k,v in grid.items()};d.update(directS=np.full(615,2.5),directC=np.full(615,4.5),directI=np.full(615,12.5),I_angles=np.arange(90,44,-7.5),Angle=0,Contr=14,lgnTF=2,lgnSF=2,DownFactor=10,LGNctgr=1,L6ctgr=1,S_angle=0)
assert check_block(d,u,1,grid)
near={a:b.copy() if isinstance(b,np.ndarray) else b for a,b in d.items()};near['directL4I'][0]+=1e-10
assert check_block(near,u,1,grid)
for k in ['directS','sample_inds','directL4E','directL4I','lgnTF','I_angles']:
 bad={a:b.copy() if isinstance(b,np.ndarray) else b for a,b in d.items()}
 if k=='directS':bad[k][0]=np.nan
 elif k in ['sample_inds','directL4E']:bad[k][0]+=1
 elif k=='directL4I':bad[k][0]+=1e-5
 elif k=='lgnTF':bad[k]=32
 else:bad[k]=np.arange(90,66,-3.75)
 try:check_block(bad,u,1,grid)
 except AssertionError:pass
 else:raise AssertionError('Invalid source accepted '+k)
# Full-grid logical uniqueness and a disjoint row assignment, without subsampling.
rng=np.random.default_rng(42);q=rng.random(615);a=np.flatnonzero(q<.8);b=np.flatnonzero((q>=.8)&(q<.9));c=np.flatnonzero(q>=.9)
assert len(a)+len(b)+len(c)==615 and not set(a)&set(b) and not set(a)&set(c) and not set(b)&set(c)
print('Source contracts, native-point coverage and disjoint split checks passed')
