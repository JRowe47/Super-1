"""Paired held-out contrasts, scale diagnostics, and construction checks."""
import copy,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def read(n):return json.loads((OUT/n).read_text())
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def estimate(values,ix,weights=None):
    a=np.asarray(values,float)
    if weights is None:point=a.mean();boot=a[ix].mean(1)
    else:
        w=np.array(weights);point=np.sum(a*w)/w.sum();boot=(a[ix]*w[ix]).sum(1)/w[ix].sum(1)
    return dict(estimate=float(point),ci95=np.quantile(boot,[.025,.975]).tolist())
def short(r,kind):return [x for x in r['retention'][kind]['sequences'] if x['n']<=48]
def contrast(a,b):
    ix=np.random.default_rng(2026101722).integers(0,48,(4000,48));out=dict(validation_nll=estimate(np.array(a['validation']['block_nll'])-b['validation']['block_nll'],ix))
    for kind in ['repeat','scrambled']:
        aa,bb=short(a,kind),short(b,kind);rng=np.random.default_rng(2026101723)
        ix=np.concatenate([rng.choice([i for i,r in enumerate(aa) if r['n']==n],(4000,16),replace=True) for n in [16,32,48]],axis=1)
        assert [r['id'] for r in aa]==[r['id'] for r in bb]
        out[kind]={k:estimate([scale*(x[k]-y[k]) for x,y in zip(aa,bb)],ix) for k,scale in [('accuracy',100),('target_nll',1),('last_query_teacher_kl',1)]}
    aa=a['retention']['prose']['passages'];bb=b['retention']['prose']['passages'];ix=np.random.default_rng(2026101724).integers(0,8,(4000,8))
    out['prose_nll']=estimate([x['nll']-y['nll'] for x,y in zip(aa,bb)],ix,[x['tokens'] for x in aa]);return out
def average_controls(a,b):
    out=copy.deepcopy(a);out['arm']='random_mean'
    out['validation']['block_nll']=((np.array(a['validation']['block_nll'])+b['validation']['block_nll'])/2).tolist();out['validation']['nll']=np.mean(out['validation']['block_nll'])
    for kind in ['repeat','scrambled']:
        for r,x,y in zip(out['retention'][kind]['sequences'],a['retention'][kind]['sequences'],b['retention'][kind]['sequences']):
            for k in ['accuracy','target_nll','last_query_teacher_kl']:r[k]=(x[k]+y[k])/2
    for r,x,y in zip(out['retention']['prose']['passages'],a['retention']['prose']['passages'],b['retention']['prose']['passages']):r['nll']=(x['nll']+y['nll'])/2
    for k in ['nll','kl_from_baseline','argmax_changed']:out['retention']['prose'][k]=(a['retention']['prose'][k]+b['retention']['prose'][k])/2
    return out
def main():
    rows=read('evaluations.json');base=rows[0];assert base['arm']=='baseline' and len(rows)==16
    for step in [1,12,24]:
        rr={r['arm']:r for r in rows if r['step']==step};rows.append(average_controls(rr['random0'],rr['random1']))
    effects=[];contrasts=[]
    for r in rows:
        item=dict(arm=r['arm'],step=r['step'],validation_nll=r['validation']['nll'],changes=contrast(r,base),prose=r['retention']['prose'])
        for kind in ['repeat','scrambled']:
            for label,sel in [(kind,short(r,kind)),(kind+'_long',[s for s in r['retention'][kind]['sequences'] if s['n']>48])]:
                item[label]={k:float(np.mean([s[k] for s in sel])) for k in ['accuracy','target_nll','last_query_teacher_kl']}
        effects.append(item)
    for step in [1,12,24]:
        rr={r['arm']:r for r in rows if r['step']==step}
        for a,b in [('targeted','random_mean'),('opposite','random_mean'),('targeted','full'),('targeted','opposite'),('random_mean','full')]:
            contrasts.append(dict(step=step,a=a,b=b,contrast=contrast(rr[a],rr[b])))
    tr=read('training.json');norms=read('reference_step_norms.json')
    maxnorm=max(abs(r['step_norm']-norms[r['step']-1]) for r in tr);assert maxnorm<1e-10
    bandrows=read('haar_update_bands.json');spectra=[]
    for arm in ['full','targeted','random0','random1','opposite']:
        rr=[r for r in bandrows if r['arm']==arm and r['step']==24];count=sum(r['count'] for r in rr);total=sum(r['update_energy'] for r in rr);assert count==18914304
        for level in sorted({r['level'] for r in rr if r['kind']=='detail'}):
            s=[r for r in rr if r['kind']=='detail' and r['level']==level];n=sum(r['count'] for r in s);e=sum(r['update_energy'] for r in s)
            spectra.append(dict(arm=arm,level=level,count=n,energy=e,enrichment=e/total/(n/count),total_update_norm=math.sqrt(total)))
    toy=read('transport_evaluations.json');toysummary=[]
    for arm in ['full12','shared2']:
        for grid in [1,6,12,17,24,31,48,96]:
            rr=[r for r in toy if r['arm']==arm and r['grid']==grid];assert len(rr)==5
            toysummary.append(dict(arm=arm,grid=grid,rotation_operations=rr[0]['rotation_operations'],**{k:dict(mean=float(np.mean([r[k] for r in rr])),min=min(r[k] for r in rr),max=max(r[k] for r in rr)) for k in ['expected_mse','empirical_mse','sign_accuracy','auxiliary_energy','max_drift_from_12_pulse']}))
    masks=read('mask_band_counts.json');assert all(len(set(r['enabled'].values()))==1 for r in masks)
    assert len(read('mask_checks.json'))==12 and all(r['frozen_coefficient_max_change']==0 for r in read('mask_checks.json'))
    checks=dict(matching_band_counts=True,matching_step_norm_max_error=maxnorm,frozen_masters_exact=True,full_resolution_count=18914304,all80_toy_evaluations_recorded=len(toy)==80,max_toy_grid_drift=max(r['max_drift_from_12_pulse'] for r in toy))
    dump('effects.json',effects);dump('contrasts.json',contrasts);dump('haar_band_summary.json',spectra);dump('transport_summary.json',toysummary);dump('analysis_checks.json',checks)
    summary=dict(baseline=effects[0],final=[r for r in effects if r['step']==24],primary=next(r for r in contrasts if r['step']==24 and r['a']=='targeted' and r['b']=='random_mean'),toy=toysummary,checks=checks)
    dump('summary.json',summary)
    print(json.dumps(dict(final=[dict(arm=r['arm'],text_nll=r['validation_nll'],copy_accuracy=r['repeat']['accuracy'],copy_nll=r['repeat']['target_nll'],long_accuracy=r['repeat_long']['accuracy'],prose_delta=r['changes']['prose_nll']['estimate']) for r in summary['final']],primary=summary['primary'],checks=checks),indent=2))
if __name__=='__main__':main()
