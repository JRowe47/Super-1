"""Read-only derivation of all preregistered comparisons; writes a new summary."""
import json,math,statistics
from pathlib import Path
H=Path(__file__).resolve().parent
C=json.loads((H/'CONFIG.json').read_text()); E=H/'execution'
rows=json.loads((E/'RESULT.json').read_text())['rows']
def row(a,s,v='selected'): return next(r for r in rows if r['arm']==a and r['seed']==s and r['variant']==v)
def contrast(a,b,bvariant='selected'):
    ds=[row(a,s)['confirmation']['ce']-row(b,s,bvariant)['confirmation']['ce'] for s in C['seeds']]
    mean=statistics.mean(ds); radius=C['t95_df2']*statistics.stdev(ds)/math.sqrt(len(ds)); upper=mean+radius
    return dict(deltas=ds,mean_ce_delta=mean,ci95=[mean-radius,upper],ppl_change_percent=100*math.expm1(mean),quality_pass=max(ds)<=math.log1p(C['quality_ppl_tolerance']) and upper<=math.log1p(C['quality_ppl_tolerance']))
summary=dict(means={},contrasts={},gates={},scope='Single-book tiny512-byte models. No SOTA,70B,100K recall, speed or total-memory claim.')
for a in C['arms']:
    rs=[row(a,s) for s in C['seeds']]
    summary['means'][a]=dict(ce=statistics.mean(r['confirmation']['ce'] for r in rs),zero_ce=statistics.mean(row(a,s,'zero')['confirmation']['ce'] for s in C['seeds']),teacher_ce=statistics.mean(row(a,s,'teacher_fp32')['confirmation']['ce'] for s in C['seeds']),payload_bytes=rs[0]['payload_bytes'],file_bytes=rs[0]['file_bytes'],selected_rounds=[r['round'] for r in rs])
for a in ('curve_b4','curve_be4','nano_q4'):
    control='nano_fp16' if a.startswith('nano') else 'curve_fp16'
    own=contrast(a,control); teacher=contrast(a,a,'teacher_fp32')
    payload=[row(control,s)['payload_bytes']/row(a,s)['payload_bytes'] for s in C['seeds']]
    files=[row(control,s)['file_bytes']/row(a,s)['file_bytes'] for s in C['seeds']]
    summary['contrasts'][a+'_vs_'+control]=own
    summary['contrasts'][a+'_vs_teacher']=teacher
    storage=min(payload+files)>=C['storage_reduction_target']
    summary['gates'][a]=dict(quality_pass=own['quality_pass'] and teacher['quality_pass'],storage_pass=storage,payload_reductions=payload,file_reductions=files,joint_pass=storage and own['quality_pass'] and teacher['quality_pass'])
for b in ('nano_fp16','nano_q4'):
    summary['contrasts']['curve_be4_vs_'+b]=contrast('curve_be4',b)
with (E/'SUMMARY.json').open('x') as f: json.dump(summary,f,indent=2,sort_keys=True); f.write('\n')
print(json.dumps(summary,indent=2))
