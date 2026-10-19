"""Paired, length-stratified effects and prespecified contrasts."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';RAW=ROOT/'responses'
def read(name):return json.loads((OUT/name).read_text())
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def indices(rows):
    rng=np.random.default_rng(2026101707)
    return np.concatenate([rng.choice(np.flatnonzero(np.array([r['n'] for r in rows])==n),(4000,sum(r['n']==n for r in rows)),replace=True) for n in sorted({r['n'] for r in rows})],axis=1)
def ci(v,ix,weights=None):
    v=np.asarray(v,np.float64)
    if weights is None:est=v.mean();samples=v[ix].mean(1)
    else:
        w=np.asarray(weights);est=np.dot(v,w)/sum(w);samples=(v[ix]*w[ix]).sum(1)/w[ix].sum(1)
    return dict(estimate=float(est),ci95=np.quantile(samples,[.025,.975]).tolist())
def values(row,kind,key='target_nll'):return np.array([r[key] for r in row[kind]['sequences']])
def row_for(rows,step,path,direction='early',lam=1):
    return next(r for r in rows if r['id']!='baseline' and r['step']==step and r['path']==path and r['direction']==direction and abs(r['lam']-lam)<1e-7)
def summarize(rows):
    effects=[];bases={r['step']:r for r in rows if r['id']=='baseline'}
    for r in rows:
        if r['id']=='baseline':continue
        b=bases[r['step']];ix=indices(b['repeat']['sequences'])
        obj={k:r[k] for k in ['id','step','split','path','direction','lam','role','intended_update_norm','actual_update_norm','rounding_l2_error']}
        g=np.load(OUT/f'logit_gram_step{r["step"]}.npy')
        with np.load(RAW/f'{r["split"]}_step{r["step"]}_baseline.npz') as z:hb={k:z[k].copy() for k in z.files}
        with np.load(RAW/f'{r["split"]}_step{r["step"]}_{r["id"]}.npz') as z:hr={k:z[k].copy() for k in z.files}
        for kind in ['repeat','scrambled']:
            assert [x['id'] for x in r[kind]['sequences']]==[x['id'] for x in b[kind]['sequences']]
            obj[kind]=dict(baseline_accuracy=b[kind]['accuracy'],accuracy=r[kind]['accuracy'],baseline_nll=b[kind]['target_nll'],nll=r[kind]['target_nll'])
            for key in ['target_nll','accuracy','first_nll','content_attention']:
                v=values(r,kind,key)-values(b,kind,key)
                if key=='accuracy':v*=100
                obj[kind][key+'_delta']=ci(v,ix)
            h=hr[kind+'_hidden'].astype(np.float64)-hb[kind+'_hidden']
            obj[kind]['centered_full_vocabulary_logit_rms']=float(np.sqrt(max(0.,np.mean(np.einsum('bi,ij,bj->b',h,g,h,optimize=True)))))
            obj[kind]['by_length']=[dict(n=n,nll_delta=float(np.mean([a['target_nll']-c['target_nll'] for a,c in zip(r[kind]['sequences'],b[kind]['sequences']) if a['n']==n]))) for n in [16,32,48]]
        pr=r['prose']['passages'];pb=b['prose']['passages'];pix=np.random.default_rng(2026101708).integers(0,len(pr),(4000,len(pr)))
        obj['prose']=dict(nll_delta=ci([a['nll']-c['nll'] for a,c in zip(pr,pb)],pix,[a['tokens'] for a in pr]),
            kl_from_baseline=r['prose']['kl_from_baseline'],argmax_changed=r['prose']['argmax_changed'],tokens=r['prose']['tokens'],
            changed_argmax_count=int(round(r['prose']['argmax_changed']*r['prose']['tokens'])))
        effects.append(obj)
    return effects
def main():
    scan=read('scan.json');con=read('confirmation.json');selection=read('selection_locked.json');meta=read('curve_metadata.json')
    se=summarize(scan);ce=summarize(con);dump('scan_effects.json',se);dump('confirmation_effects.json',ce)
    path=selection['prior_selected'];contrasts=[]
    b=scan[0];target=row_for(scan,1000,path);ix=indices(b['repeat']['sequences'])
    for kind in ['repeat','scrambled']:
        other=np.stack([values(r,kind) for r in scan[1:] if r['path']!=path]).mean(0)
        for name,v in [('selected_minus_other63_mean',values(target,kind)-other),('selected_minus_sibling',values(target,kind)-values(row_for(scan,1000,'aaaaaa'),kind))]:
            contrasts.append(dict(split='scan',probe=kind,contrast=name,**ci(v,ix)))
    bases={r['step']:r for r in con if r['id']=='baseline'};ix=indices(bases[1000]['repeat']['sequences'])
    def effect(step,direction,kind):return values(row_for(con,step,path,direction),kind)-values(bases[step],kind)
    for kind in ['repeat','scrambled']:
        for p in selection['paths']:
            if p==path:continue
            v=values(row_for(con,1000,path),kind)-values(row_for(con,1000,p),kind)
            contrasts.append(dict(split='confirmation',probe=kind,contrast='selected_minus_'+p,**ci(v,ix)))
        for direction in ['early','late']:
            contrasts.append(dict(split='confirmation',probe=kind,contrast=direction+'_context8000_minus1000',**ci(effect(8000,direction,kind)-effect(1000,direction,kind),ix)))
        for step in [512,1000,8000]:
            contrasts.append(dict(split='confirmation',probe=kind,contrast=f'step{step}_late_minus_early',**ci(effect(step,'late',kind)-effect(step,'early',kind),ix)))
        did=(effect(8000,'late',kind)-effect(1000,'late',kind))-(effect(8000,'early',kind)-effect(1000,'early',kind))
        contrasts.append(dict(split='confirmation',probe=kind,contrast='direction_by_context_difference_in_differences',**ci(did,ix)))
    dump('paired_contrasts.json',contrasts)
    curvature=[]
    for step in [512,1000,8000]:
        for direction in ['early','late']:
            for kind in ['repeat','scrambled']:
                pos=values(row_for(con,step,path,direction,1),kind);neg=values(row_for(con,step,path,direction,-1),kind)
                v=pos+neg-2*values(bases[step],kind)
                curvature.append(dict(step=step,direction=direction,probe=kind,quantity='f(+1)+f(-1)-2f(0)',**ci(v,ix)))
    dump('dose_curvature.json',curvature)
    ranks={}
    for kind in ['repeat','scrambled']:
        ordered=sorted(se,key=lambda r:r[kind]['target_nll_delta']['estimate'],reverse=True)
        vals=[r[kind]['target_nll_delta']['estimate'] for r in ordered]
        ranks[kind]=dict(selected_rank=1+next(i for i,r in enumerate(ordered) if r['path']==path),total=64,winner=ordered[0]['path'],
            minimum=float(min(vals)),median=float(np.median(vals)),maximum=float(max(vals)),
            selected_nll_delta=next(r for r in ordered if r['path']==path)[kind]['target_nll_delta'],
            ordered_paths=[r['path'] for r in ordered])
    allrows=scan[1:]+[r for r in con if r['id']!='baseline']
    checks=dict(all_parameters_restored=all(r['restored_exactly'] for r in allrows),any_change_outside_window=any(r['changed_outside_window'] for r in allrows),
        max_rounding_l2_error=max(r['rounding_l2_error'] for r in allrows),max_rounding_coordinate_error=max(r['rounding_max_abs_error'] for r in allrows),
        max_relative_norm_error=max(abs(r['actual_update_norm']/r['intended_update_norm']-1) for r in allrows))
    summary=dict(curves=meta,selection=selection,scan_ranks=ranks,paired_contrasts=contrasts,
        baselines=[dict(step=r['step'],repeat_accuracy=r['repeat']['accuracy'],scrambled_accuracy=r['scrambled']['accuracy'],prose_nll=r['prose']['nll'],prose_tokens=r['prose']['tokens']) for r in con if r['id']=='baseline'],
        selected_unit_rollback=[r for r in ce if r['path']==path and r['lam']==1],verification=checks,environment=read('environment.json'))
    dump('summary.json',summary)
    print(json.dumps(dict(scan_ranks=ranks,paired_contrasts=contrasts,selected_unit_rollback=[dict(step=r['step'],direction=r['direction'],repeat=r['repeat']['target_nll_delta'],scrambled=r['scrambled']['target_nll_delta'],prose=r['prose']) for r in summary['selected_unit_rollback']],verification=checks),indent=2))
if __name__=='__main__':main()
