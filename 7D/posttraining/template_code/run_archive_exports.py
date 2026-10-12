from pathlib import Path
import hashlib,json,os,signal,subprocess,time

P=Path(__file__).resolve().parent.parent
C=P/'code'
marker=P/'archive_matlab_saved.txt'
assert not marker.exists(),'Reconcile prior saved export before retry'
command="addpath('"+str(C)+"');export_archive_curves_layout('"+str(P)+"');export_comparisons('"+str(P)+"');fid=fopen('"+str(marker)+"','w');fwrite(fid,'complete');fclose(fid)"
proc=subprocess.Popen(['matlab','-singleCompThread','-batch',command],start_new_session=True)
saved=None
bounded=False
while proc.poll() is None:
    if marker.is_file():
        if saved is None:saved=time.monotonic()
        if time.monotonic()-saved>60:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            bounded=True
            break
    time.sleep(5)
assert marker.is_file(),('MATLAB failed before saving',proc.returncode)
if saved is None:assert proc.returncode==0,proc.returncode
count=len(json.loads((P/'representative_reference_manifest.json').read_text()))
files=list((P/'local_archive').glob('*.pdf'))+list((P/'local_archive/HC norm').glob('*.pdf'))
assert len(files)==4+count,(len(files),count)
for f in files:assert f.read_bytes().startswith(b'%PDF') and f.stat().st_size>1000
(P/'remote_archive_complete.json').write_text(json.dumps({'PDF_count':len(files),'PDFs':{str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'bounded_cleanup':bounded,'local_visual_QA_complete':False,'email_sent':False},indent=2))
print('REMOTE_ARCHIVE_SAVED',len(files),str(P))
