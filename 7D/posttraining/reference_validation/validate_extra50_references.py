"""Check the complete reusable extra50 library-equilibrium deliverable."""
import csv
import hashlib
import json

import numpy as np
from scipy.io import loadmat
from prepare_extra50_full import ROOT

with (ROOT / 'extra50_cases.tsv').open() as f:
    cases = list(csv.DictReader(f, delimiter='\t'))
assert len(cases) == 50
records = []
for case in cases:
    folder = ROOT / case['tag'] / 'equilibrium_library'
    angle, contrast = float(case['angle']), int(case['contrast'])
    mat = folder / f'libonly_16HC_6DprefAngle_input_contr{contrast}_angle_{angle:.2f}.mat'
    png = folder / f'LibraryOnly_angle_{angle:.2f}_contr{contrast}.png'
    fields = ['Angle', 'Contr', 'lgnTF', 'lgnSF', 'LDEEquv',
              'S_map_fulllib', 'C_map_fulllib', 'I_map_fulllib', 'E_map_fulllib', 'EpocTest']
    data = loadmat(mat, variable_names=fields)
    for field, key in [('Angle', 'angle'), ('Contr', 'contrast'), ('lgnTF', 'tf'), ('lgnSF', 'sf')]:
        assert float(np.asarray(data[field]).item()) == float(case[key]), (mat, field)
    assert int(np.asarray(data['EpocTest']).item()) == 300
    assert 'LDEEquv' in data
    for field in fields[5:9]:
        assert data[field].shape == (40, 40) and np.isfinite(data[field]).all(), (mat, field)
    np.testing.assert_allclose(data['E_map_fulllib'],
                               .6923*data['S_map_fulllib']+.3077*data['C_map_fulllib'])
    with png.open('rb') as f:
        assert f.read(8) == b'\x89PNG\r\n\x1a\n', png
    records.append(dict(case=int(case['case']), mat=str(mat), png=str(png),
                        mat_sha256=hashlib.sha256(mat.read_bytes()).hexdigest()))
result = dict(status='complete', count=len(records), references=records)
(ROOT / 'references_ready.json').write_text(json.dumps(result, indent=2)+'\n')
print('Verified all 50 library equilibrium MAT files and PNG headers.')
