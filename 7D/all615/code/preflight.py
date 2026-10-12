from pathlib import Path
import json,subprocess,os
import h5py,numpy as np
from source_bundle import sha
R=Path(__file__).resolve().parents[1]
original=(R/'code/source_simulator_original.m').read_text()
body=original[original.index("S = load(sprintf"):original.index('if DirectRowsOnly\n    sample_inds = selectSpaceFillingIndices')]
header="""repo='/scratch/xh2906/NYU-Vision-2Drive-main'; cd(repo); addpath(repo,fullfile(repo,'Utils'),fullfile(repo,'Data'),fullfile(repo,'MFpVLDEComput'),fullfile(repo,'Data','Paper2_NetworkTuning','Fig1V4')); DataPt='V4D2'; Angle=0; L6low=3; L6up=120; DownFactor=10;
"""
footer=f"""valid=find(isfinite(L4ERcrd)&isfinite(L4IRcrd)&L4ERcrd>=0&L4IRcrd>=0); assert(numel(valid)==615); sample_inds=valid; directL4E=L4ERcrd(valid); directL4I=L4IRcrd(valid); save('{R}/grid_native.mat','sample_inds','directL4E','directL4I','nE','nI','Step','LineFit','-v7.3');"""
p=Path(os.environ['TMPDIR'])/'grid_preflight.m';p.write_text(header+body+footer)
subprocess.run(['matlab','-batch',f"run('{p}')"],check=True)
with h5py.File(R/'grid_native.mat') as f:d={k:f[k][:].ravel().tolist() for k in ['sample_inds','directL4E','directL4I','nE','nI','Step','LineFit']}
assert len(d['sample_inds'])==615 and len(set(zip(d['directL4E'],d['directL4I'])))==615
assert np.isfinite(d['directL4E']).all() and np.isfinite(d['directL4I']).all()
d.update(verified=True,native_valid_points=615,grid_sha256=sha(R/'grid_native.mat'),original_simulator_sha256=sha(R/'code/source_simulator_original.m'),run_simulator_sha256=sha(R/'code/run_ds10_all615.m'))
(R/'grid_validation.json').write_text(json.dumps(d,indent=2))
print('NATIVE_DS10_GRID_VERIFIED',615,flush=True)
