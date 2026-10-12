"""Read-only learning evidence used by monitoring and the terminal post-training gate."""
from pathlib import Path
import csv,json,math
def assess(root):
 d=Path(root)/'train/C'
 rows=list(csv.DictReader((d/'train_log.csv').open())) if (d/'train_log.csv').exists() else []
 gradients=list(csv.DictReader((d/'gradcos_log.csv').open())) if (d/'gradcos_log.csv').exists() else []
 initial=json.loads((d/'initial_validation_metrics.json').read_text()) if (d/'initial_validation_metrics.json').exists() else {}
 finite_rows=[r for r in rows if r.get('epoch') and r.get('val_total_loss')]
 post_initial=[r for r in gradients if int(float(r['global_step']))>1]
 recent=post_initial[-3:]
 terms=['rmse_term__grad_norm','relrmse_raw__grad_norm','tf_unimodal_loss__grad_norm','sf_unimodal_loss__grad_norm']
 def zero(r):return all(r.get(k) not in (None,'') and math.isfinite(float(r[k])) and float(r[k])==0 for k in terms)
 epoch=int(float(finite_rows[-1]['epoch'])) if finite_rows else 0
 finite=all(math.isfinite(float(r['val_total_loss'])) and math.isfinite(float(r['val_rmse_raw'])) for r in finite_rows)
 minimum=min((float(r['val_total_loss']) for r in finite_rows),default=None)
 initial_loss=initial.get('total_loss')
 improved=bool(minimum is not None and initial_loss is not None and minimum<float(initial_loss))
 last_losses=[float(r['val_total_loss']) for r in finite_rows[-3:]]
 stagnant=len(last_losses)==3 and all(math.isfinite(v) for v in last_losses) and max(last_losses)-min(last_losses)<=1e-8
 collapse=epoch>=10 and stagnant and len(recent)==3 and all(zero(r) for r in recent)
 active_gradient=bool(recent and any(any(r.get(k) not in (None,'') and math.isfinite(float(r[k])) and float(r[k])>0 for k in terms[:2]) for r in recent))
 return dict(last_epoch=epoch,initial_validation_loss=initial_loss,lowest_logged_validation_loss=minimum,validation_improved=improved,finite_logged_metrics=finite,last_gradient_steps=[r['global_step'] for r in recent],persistent_zero_gradients=collapse,active_recent_fit_gradient=active_gradient,normal_learning=bool(improved and finite and active_gradient and not collapse),last_row=finite_rows[-1] if finite_rows else None,gradient_rows=recent)

