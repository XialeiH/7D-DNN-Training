from pathlib import Path
import json,os,shlex,sys,traceback
import h5py,numpy as np
from source_bundle import check_block
from source_completion import verify_saved_bundle,finish_saved_bundle,run_until_saved
R=Path(__file__).resolve().parents[1]
unit=int(json.loads(Path(sys.argv[2]).read_text())['unit_ids'][int(sys.argv[3])]) if sys.argv[1]=='--wave' else int(sys.argv[1])
pilot='--pilot' in sys.argv
u=json.loads((R/'source_units.json').read_text())[unit];grid=json.loads((R/'grid_validation.json').read_text())
bundle=R/'source_bundles'/f'unit_{unit:04d}.h5';tmp=Path(os.environ['TMPDIR']);tmp.mkdir(parents=True,exist_ok=True)
try:
 with h5py.File(bundle,'a') as f:
  if '_writing' in f:del f['_writing']
  done=[]
  for l6 in range(1,41):
   key=f'L6_{l6:02d}'
   if key in f:
    g=f[key];assert g.attrs['verified'];check_block({k:g[k][()] for k in g},u,l6,grid);done.append(l6)
  f.attrs['unit_json']=json.dumps(u);f.attrs['status']='running';f.flush()
 wanted=[1,40] if pilot else list(range(1,41));missing=[l6 for l6 in wanted if l6 not in done]
 if not missing:
  verify_saved_bundle(bundle,u,wanted,grid,deep=True);finish_saved_bundle(bundle,pilot)
  print('SOURCE_DATA_COMPLETE; no MATLAB needed',unit,flush=True);sys.exit(0)
 driver=[f"addpath('{R}/code','-begin');"]
 for l6 in missing:
  reuse=u['reuse_blocks'][l6-1];block=l6+(160 if u['background'] else 0)
  out=tmp/'matlab_output'
  driver += [f"setenv('H96_REUSE_BLOCK','{reuse}');",f"run_ds10_all615({u['angle']},{u['contrast']},{u['tf']},{u['sf']},10,'{out}',{block},615,true);",f"files=dir(fullfile('{out}','blocks_*','block_*.mat')); assert(numel(files)==1); p=fullfile(files(1).folder,files(1).name);"]
  base=shlex.join(['env','-u','LD_LIBRARY_PATH',sys.executable,str(R/'code/source_bundle.py'),str(unit),str(l6)])
  driver += [f"cmd=sprintf('{base} \"%s\" \"{reuse}\"',p); [status,msg]=system(cmd); disp(msg); assert(status==0,'Source bundle validation/commit failed'); delete(p);"]
 marker=tmp/'source_data_saved';marker.unlink(missing_ok=True)
 driver += [f"fid=fopen('{marker}','w'); assert(fid>=0); fprintf(fid,'all requested source blocks committed'); fclose(fid);", "pool=gcp('nocreate'); if ~isempty(pool),delete(pool);end"]
 (tmp/'driver.m').write_text('\n'.join(driver))
 log=tmp/'matlab.log'
 def accept_saved():
  verify_saved_bundle(bundle,u,wanted,grid,deep=True);finish_saved_bundle(bundle,pilot)
 returncode,forced=run_until_saved(['matlab','-batch',f"run('{tmp}/driver.m')"],log,marker,accept_saved)
 with h5py.File(bundle,'a') as f:
  if 'matlab_log' in f:del f['matlab_log']
  f.create_dataset('matlab_log',data=np.frombuffer(log.read_bytes(),dtype=np.uint8),compression='gzip')
  f.attrs['matlab_returncode']=returncode;f.attrs['matlab_cleanup_forced_after_saved']=forced
  f.attrs['status']='pilot_verified' if pilot else 'complete';f.attrs['complete']=not pilot;f.flush()
except Exception:
 with h5py.File(bundle,'a') as f:
  if not f.attrs.get('complete',False):f.attrs['status']='failed';f.attrs['error']=traceback.format_exc();f.flush()
 raise
