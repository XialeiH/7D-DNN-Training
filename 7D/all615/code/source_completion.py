"""Accept validated source data independently of MATLAB shutdown status."""
import json, os, signal, subprocess, time
import h5py
from source_bundle import check_block


def verify_saved_bundle(path, unit, wanted, grid=None, deep=False):
    with h5py.File(path, 'r') as f:
        assert json.loads(f.attrs['unit_json']) == unit, 'Source-unit identity mismatch'
        assert '_writing' not in f, 'Uncommitted source block remains'
        for n in wanted:
            g = f[f'L6_{n:02d}']
            assert g.attrs['verified'], ('Unverified block', n)
            assert len(g.attrs['raw_mat_sha256']) == 64, ('Missing raw source hash', n)
            if deep:
                check_block({k: g[k][()] for k in g}, unit, n, grid)


def finish_saved_bundle(path, pilot=False):
    with h5py.File(path, 'a') as f:
        f.attrs['status'] = 'pilot_verified' if pilot else 'complete'
        f.attrs['complete'] = not pilot
        f.attrs['completion_policy'] = 'all requested native blocks validated; MATLAB exit status is secondary'
        if 'error' in f.attrs:
            del f.attrs['error']
        f.flush()


def run_until_saved(command, log, marker, on_saved, grace=60, interval=2):
    """Bound only cleanup AFTER on_saved has verified the requested data."""
    validated = False
    deadline = None
    forced = False
    with open(log, 'w') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            if not validated and marker.exists():
                try:
                    on_saved()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                validated = True
                deadline = time.monotonic() + grace
            if validated and time.monotonic() >= deadline:
                forced = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(interval)
        if not validated:
            on_saved()  # Accept abnormal exits only when all data actually passes validation.
    return process.returncode, forced
