from pathlib import Path
import csv, datetime, hashlib, json, sys
import numpy as np
from scipy.io import loadmat

trials = {'seed43': {'root': '/home/xh2906/librarySCI_C_seed_retry/h96_tfsf7d_h1024x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20261004_original_loss_relu_Cseed43_20261005'}}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and all(same(x, y) for x, y in zip(a.flat, b.flat))
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return bool(np.isclose(a, b, rtol=0, atol=1e-12))

report = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runs': {}}
for label in (sys.argv[1:] or list(trials)):
    p = Path(trials[label]['root']) / 'posttraining_7d_tf32_globalall7'
    a = {'post_root': str(p), 'atlas': [], 'atlas_errors': [], 'dense': [], 'representatives': [], 'errors': []}
    cp = json.loads((p / 'checkpoint_provenance.json').read_text())
    runtime = json.loads((p / 'atlas_139/runtime_provenance.json').read_text())
    for t in 'SCI':
        assert cp[t]['sha256'] == runtime['models'][t]['sha256'] == sha(p / 'checkpoints' / (t + '.pt'))
    a['checkpoint_epochs'] = {t: cp[t]['extra']['best_epoch'] for t in 'SCI'}
    rows = list(csv.DictReader((p / 'atlas_139/cases.tsv').open(), delimiter='\t'))
    assert len(rows) == 139 and all(float(row['sf']) <= 10 for row in rows)
    csv_path = p / 'atlas_139/hcnorm_all_139.csv'
    recorded = {int(row['case']): row for row in csv.DictReader(csv_path.open())} if csv_path.exists() else {}
    for row in rows:
        mats = list(Path(row['save_dir']).glob('eig_16HC_*.mat'))
        if not mats:
            a['atlas_errors'].append({'case': row['case'], 'error': 'missing saved MAT'})
            continue
        try:
            assert len(mats) == 1
            f = mats[0]
            m = loadmat(f, simplify_cells=True)
            trace = np.asarray(m['err_ode_true_all'])
            assert trace.ndim == 2 and trace.shape[1] >= 300 and np.isfinite(trace).all()
            assert all(np.isfinite(m[k + suffix]).all() for k in 'SCI' for suffix in ['_map_full','_map_fulllib'])
            assert np.array_equal([m[k] for k in ['Angle','Contr','lgnTF','lgnSF']], [float(row[k]) for k in ['angle','contrast','tf','sf']])
            diagnostics={}
            for field in ['OperatingTarget','OperatingPointLib']:
                if field+'Valid' in m:
                    valid=bool(m[field+'Valid']);diagnostics[field+'Valid']=valid
                    if not valid: assert np.size(m[field])==0
            hc = float(np.mean(trace[-1, 250:]))
            if recorded:
                assert np.isclose(hc, float(recorded[int(row['case'])]['HCnorm']), rtol=0, atol=1e-9)
            a['atlas'].append({'case': int(row['case']), 'HCnorm': hc, 'trace_columns': trace.shape[1], 'sha256': sha(f), 'optional_diagnostics': diagnostics})
        except Exception as e:
            a['atlas_errors'].append({'case': row['case'], 'error': type(e).__name__ + ': ' + str(e)})
    if len(a['atlas']) == 139 and not a['atlas_errors'] and recorded:
        values = [v['HCnorm'] for v in a['atlas']]
        calculated = {'count': 139, 'mean': float(np.mean(values)), 'min': min(values), 'max': max(values), 'HC_gt_5': sum(v > 5 for v in values)}
        summary = json.loads((p / 'atlas_139/hcnorm_summary.json').read_text())
        assert len(recorded) == 139
        for k, v in calculated.items():
            assert np.isclose(v, summary[k], rtol=0, atol=1e-9)
        a['atlas_summary_verified'] = calculated
    refs = json.loads((p / 'representative_reference_manifest.json').read_text())
    baseline = loadmat(refs[0]['reference'], simplify_cells=True)
    tf_maps = {}
    for row in csv.DictReader((p / 'dense_tuning/cases.tsv').open(), delimiter='\t'):
        f = p / 'dense_tuning/equilibrium' / ('case_%03d.mat' % int(row['task']))
        try:
            m = loadmat(f, simplify_cells=True)
            q = [float(row[k]) for k in ['angle', 'contrast', 'tf', 'sf']]
            assert m['EpocTest'] == 300 and np.array_equal([m[k] for k in ['angle', 'Contr', 'lgnTF', 'lgnSF']], q)
            assert all(np.isfinite(m[k + '_map_full']).all() for k in 'SCI') and np.isfinite(m['final_step_max'])
            for k in ['p', 'EKpUse', 'IKpUse']:
                assert same(m[k], baseline[k]), k
            a['dense'].append({'task': int(row['task']), 'condition': q, 'final_step_max': float(m['final_step_max']), 'sha256': sha(f)})
            if q[0] == 0 and q[1] == 100 and q[3] == 2.5 and q[2] in [24, 32]:
                tf_maps[q[2]] = {str(pix): {k: float(m[k + '_map_full'][pix[0] - 1, pix[1] - 1]) for k in 'SCI'} for pix in [(5, 10), (1, 10)]}
        except Exception as e:
            a['errors'].append({'kind': 'dense', 'task': row['task'], 'error': type(e).__name__ + ': ' + str(e)})
    comparison_file=p/'local_archive/HC norm/comparisons.csv'
    comparisons={int(r['case']):r for r in csv.DictReader(comparison_file.open())} if comparison_file.exists() else {}
    for ref in refs:
        try:
            i = ref['case']
            lp, mp = Path(ref['reference']), Path(ref['maps'])
            assert sha(lp) == ref['reference_sha256'] and sha(mp) == ref['maps_sha256']
            lib, maps = loadmat(lp, simplify_cells=True), loadmat(mp, simplify_cells=True)
            f = p / 'representative192/data' / f'case_{i:03d}.mat'
            nn = loadmat(f, simplify_cells=True)
            q = [ref[k] for k in ['orientation', 'contrast', 'tf', 'sf']]
            assert nn['EpocTest'] == 300 and np.array_equal(maps['repQ'], q)
            assert np.array_equal([nn[k] for k in ['angle', 'Contr', 'lgnTF', 'lgnSF']], q)
            for k in ['p', 'EKpUse', 'IKpUse']:
                assert same(nn[k], lib[k]), k
            assert same(nn['L6ParamUse'], lib['L6pars'][lib['L6parId'] - 1])
            l = np.stack([lib[k + '_map_fulllib'] for k in 'SCI'], axis=2)
            n = np.stack([nn[k + '_map_full'] for k in 'SCI'], axis=2)
            assert np.array_equal(l, maps['repMaps']) and np.isfinite(l).all() and np.isfinite(n).all()
            le, ne = .6923 * l[:, :, 0] + .3077 * l[:, :, 1], .6923 * n[:, :, 0] + .3077 * n[:, :, 1]
            hc = float(np.sqrt(np.mean(.8 * (le - ne) ** 2 + .2 * (l[:, :, 2] - n[:, :, 2]) ** 2)))
            if comparisons:
                assert np.isclose(hc,float(comparisons[i]['HCnorm']),rtol=0,atol=1e-8)
                for t in 'SCI': assert comparisons[i][t+'_checkpoint_sha256']==cp[t]['sha256']
            a['representatives'].append({'case': i, 'HCnorm_final_state': hc, 'final_step_max': float(nn['final_step_max']), 'NN_sha256': sha(f), 'reference_sha256': sha(lp)})
        except Exception as e:
            a['errors'].append({'kind': 'representative', 'case': ref['case'], 'error': type(e).__name__ + ': ' + str(e)})
    for frequencies in tf_maps.values():
        for pixel in frequencies.values():
            pixel['E'] = .6923 * pixel['S'] + .3077 * pixel['C']
    a['TF24_vs_TF32'] = tf_maps
    a['unsettled_dense'] = [v for v in a['dense'] if v['final_step_max'] > .01]
    a['saved_dense_and_representatives_verified'] = len(a['dense']) == 97 and len(a['representatives']) == len(refs) and not a['errors']
    report['runs'][label] = a
print(json.dumps(report, indent=2))
