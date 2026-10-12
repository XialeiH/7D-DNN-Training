import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from PIL import Image
root=Path(sys.argv[1]); rows=list(csv.DictReader((root/'cases.tsv').open(),delimiter='\t'))
out=root/'plots';out.mkdir(exist_ok=True)
for r in rows:
 d=loadmat(root/'equilibrium'/('case_%03d.mat'%int(r['task'])))
 for key in ['contrast','angle','tf','sf']:r[key]=float(r[key])
 r['final_step_max']=float(d['final_step_max'].item())
 for row,col in [(5,10),(1,10)]:
  v=[float(d[k+'_map_full'][row-1,col-1]) for k in ['S','C','I']]
  v.append(.6923*v[0]+.3077*v[1]);r[f'pixel{row}_{col}']=v
 assert np.isfinite(r['final_step_max']) and all(np.isfinite(r[k]).all() for k in ['pixel5_10','pixel1_10'])
summary={'cases':len(rows),'max_final_step':max(r['final_step_max'] for r in rows),'conditions_final_step_gt_0p01':[r for r in rows if r['final_step_max']>.01], 'hcnorm_valid':False,'tf24_vs_tf32':{}}
names=['S','C','I','E']
def select(kind,angle=0):return sorted([r for r in rows if r['kind']==kind and r['angle']==angle],key=lambda r:r['contrast' if kind=='contrast' else kind])
def draw(ax,rs,kind,pixel,ch):
 x=[r['contrast' if kind=='contrast' else kind] for r in rs]
 if kind=='contrast':x=PchipInterpolator([0,19,42,66,100],[0,4,10,25,64])(x)
 ax.plot(x,[r[pixel][ch] for r in rs],'-o',ms=2,lw=1.3,label=pixel)
 ax.set_xscale('linear')
 if kind=='contrast':ax.set_yscale('linear')
 ax.set_xlabel('real contrast C' if kind=='contrast' else kind.upper());ax.set_ylabel('firing rate (Hz)');ax.set_title(names[ch]);ax.grid(alpha=.25)
def save(fig,stem):
 fig.tight_layout();fig.savefig(out/(stem+'.png'),dpi=200);plt.close(fig)
 with Image.open(out/(stem+'.png')) as im:im.load();assert im.width>0
for kind,angle,pixel,stem in [('contrast',0,'pixel5_10','contrast_tuning_angle0_pixel5_10'),('contrast',22.5,'pixel1_10','contrast_tuning_angle22p5_pixel1_10')]+[(kind,0,pixel,f'{kind}_tuning_angle0_contr100_{pixel}') for kind in ['tf','sf'] for pixel in ['pixel5_10','pixel1_10']]:
 rs=select(kind,angle);fig,axs=plt.subplots(2,2,figsize=(10,7))
 for ch,ax in enumerate(axs.flat):draw(ax,rs,kind,pixel,ch)
 fig.suptitle(f'Dense 7D DNN equilibrium | {kind}, angle {angle:g}, {pixel}')
 save(fig,stem)
 with (out/(stem+'.csv')).open('w') as f:
  w=csv.writer(f);w.writerow(['contrast','angle','tf','sf','S','C','I','E','final_step_max'])
  for r in rs:w.writerow([r[k] for k in ['contrast','angle','tf','sf']]+r[pixel]+[r['final_step_max']])
fig,axs=plt.subplots(2,4,figsize=(16,8))
for i,kind in enumerate(['tf','sf']):
 for ch in range(4):
  for pixel in ['pixel5_10','pixel1_10']:draw(axs[i,ch],select(kind),kind,pixel,ch)
  axs[i,ch].legend()
save(fig,'tfsf_tuning_angle0_contr100_combined_pixels')
for pixel in ['pixel5_10','pixel1_10']:
 summary['tf24_vs_tf32'][pixel]={str(tf):next(r[pixel] for r in select('tf') if r['tf']==tf) for tf in [24,32]}
(root/'dense_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'dense_tuning_complete.txt').write_text('All cases finite; seven PNGs decoded. See final_step diagnostics for convergence.\n')
