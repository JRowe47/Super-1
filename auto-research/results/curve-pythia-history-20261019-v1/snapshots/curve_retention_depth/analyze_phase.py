"""Paired estimates, full-resolution Haar summaries, and an exact depth-limit diagnostic."""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def read(n):return json.loads((OUT/n).read_text())
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def estimate(values,idx,weights=None):
    a=np.asarray(values,float)
    if weights is None:point=a.mean();boot=a[idx].mean(1)
    else:
        w=np.asarray(weights,float);point=(a*w).sum()/w.sum();boot=(a[idx]*w[idx]).sum(1)/w[idx].sum(1)
    return dict(estimate=float(point),ci95=np.quantile(boot,[.025,.975]).tolist())
def bootstrap_groups(rows):
    rng=np.random.default_rng(2026101750)
    return np.concatenate([rng.choice([i for i,r in enumerate(rows) if r['n']==n],(4000,sum(r['n']==n for r in rows)),replace=True) for n in sorted({r['n'] for r in rows})],1)
def short(r,kind):return [x for x in r['retention'][kind]['sequences'] if x['n']<=48]
def contrast(a,b):
    ix=np.random.default_rng(2026101751).integers(0,24,(4000,24));out={'validation_nll':estimate(np.array(a['validation']['block_nll'])-b['validation']['block_nll'],ix)}
    for kind in ['repeat','scrambled']:
        aa,bb=short(a,kind),short(b,kind);ix=bootstrap_groups(aa)
        out[kind]={key:estimate([scale*(x[key]-y[key]) for x,y in zip(aa,bb)],ix) for key,scale in [('accuracy',100),('target_nll',1),('last_query_teacher_kl',1)]}
    aa=a['retention']['prose']['passages'];bb=b['retention']['prose']['passages'];ix=np.random.default_rng(2026101752).integers(0,8,(4000,8))
    out['prose_nll']=estimate([x['nll']-y['nll'] for x,y in zip(aa,bb)],ix,[x['tokens'] for x in aa]);return out
def main():
    rows=read('evaluations.json');baseline=next(r for r in rows if r['arm']=='baseline');effects=[]
    for r in rows:
        e=dict(arm=r['arm'],step=r['step'],validation_nll=r['validation']['nll'],changes=contrast(r,baseline),prose={k:r['retention']['prose'][k] for k in ['nll','kl_from_baseline','argmax_changed','tokens']})
        for kind in ['repeat','scrambled']:
            selected=short(r,kind);long=[x for x in r['retention'][kind]['sequences'] if x['n']>48]
            e[kind]={k:float(np.mean([a[k] for a in selected])) for k in ['accuracy','target_nll','last_query_teacher_kl']}
            e[kind+'_long']={k:float(np.mean([a[k] for a in long])) for k in ['accuracy','target_nll','last_query_teacher_kl']}
        effects.append(e)
    finals={r['arm']:r for r in rows if r['step']==24};comparisons=[]
    for coord in ['native','curve']:
        for lam in [1,4]:
            a=f'{coord}_lambda{lam}';b=f'{coord}_lambda0';comparisons.append(dict(a=a,b=b,contrast=contrast(finals[a],finals[b])))
    for lam in [0,1,4]:
        a=f'curve_lambda{lam}';b=f'native_lambda{lam}';comparisons.append(dict(a=a,b=b,contrast=contrast(finals[a],finals[b])))
    dump('effects.json',effects);dump('contrasts.json',comparisons)
    bb=[r for r in read('haar_update_bands.json') if r['step']==24];spectra=[];layerbands=[]
    for arm in sorted({r['arm'] for r in bb}):
        armrows=[r for r in bb if r['arm']==arm];total=sum(r['update_energy'] for r in armrows);count=sum(r['count'] for r in armrows)
        assert count==18914304
        for layer in [None,0,1,2,3,4,5]:
            selected=armrows if layer is None else [r for r in armrows if r['layer']==layer]
            for kind,lev in sorted({(r['kind'],r['level']) for r in selected}):
                group=[r for r in selected if r['kind']==kind and r['level']==lev];energy=sum(r['update_energy'] for r in group);n=sum(r['count'] for r in group)
                obj=dict(arm=arm,layer=layer,kind=kind,level=lev,count=n,update_energy=energy,mean_squared_coefficient=energy/n,energy_fraction=energy/total,parameter_fraction=n/count,enrichment=(energy/total)/(n/count),total_update_norm=math.sqrt(total))
                (spectra if layer is None else layerbands).append(obj)
    dump('haar_band_summary.json',spectra);dump('haar_layer_band_summary.json',layerbands)
    trajectories=read('transport_trajectories.json');toy=[]
    for arm in sorted({r['arm'] for r in trajectories}):
        last=600 if arm in ['native6','curve6','curve2'] else 400;rr=sorted([r for r in trajectories if r['arm']==arm and r['step']==last],key=lambda r:r['seed']);assert len(rr)==5
        for n in [3,4,6,8,12,16,24,48]:
            vals=[next(a for a in r['metrics'] if a['depth']==n) for r in rr];item=dict(arm=arm,depth=n,seeds=[r['seed'] for r in rr],parameters=rr[0]['active_parameters'],stages=rr[0]['cumulative_stages'])
            for k in ['expected_mse','empirical_mse','sign_accuracy','gaussian_sign_accuracy','expected_output_difference_from_depth6']:
                v=[a[k] for a in vals];item[k]=dict(mean=float(np.mean(v)),min=float(min(v)),max=float(max(v)),values=v)
            toy.append(item)
    dump('transport_summary.json',toy)
    # Continuous-limit operator for the constant oracle, using a convergent Poisson series.
    eye=np.eye(12);shift=np.roll(eye,-1,axis=1);target=np.linalg.matrix_power(shift,6);op=np.zeros_like(eye);power=eye.copy();weight=math.exp(-6)
    for k in range(100):
        if k:weight*=6/k;power=power@shift
        op+=weight*power
    mse=float(((op-target)**2).sum()/12);energy=float((op*op).sum()/12);cov=float((op*target).sum()/12)
    oracle=read('transport_oracle.json');frequency=[]
    for row in oracle:
        response=np.fft.fft(np.array(row['operator'])[0]);frequency.append(dict(depth=row['depth'],magnitudes=np.abs(response).tolist(),phases=np.angle(response).tolist()))
    limit=dict(expected_mse=mse,gaussian_sign_accuracy=1-math.acos(cov/math.sqrt(energy))/math.pi,operator=op.tolist(),fourier_magnitudes=np.abs(np.fft.fft(op[0])).tolist(),series_total_probability=float(op[0].sum()),finite_frequency_responses=frequency,
        proof='For any real scalar depth field a(s)I+b(s)S, generators commute and the exact flow is exp(A I+B S). Matching S^6 requires unit magnitude on every Fourier mode. Modes0 and6 force A+B=0 and A-B=0, hence A=B=0 and identity, contradicting S^6. This proves non-representability only in this narrow one-hop scalar continuous-flow family; it does not apply to augmented-state or general GSA architectures.')
    assert abs(op[0].sum()-1)<1e-12;dump('transport_continuous_limit.json',limit)
    summary=dict(final=[r for r in effects if r['step']==24],baseline=next(r for r in effects if r['arm']=='baseline'),contrasts=comparisons,toy=toy,continuous_limit={k:v for k,v in limit.items() if k in ['expected_mse','gaussian_sign_accuracy','proof']},pythia_runtime=read('environment.json'),toy_runtime=read('transport_environment.json'))
    dump('summary.json',summary)
    print(json.dumps(dict(pythia=[dict(arm=r['arm'],nll=r['validation_nll'],copy=r['repeat']['accuracy'],long_copy=r['repeat_long']['accuracy'],prose_delta=r['changes']['prose_nll']) for r in effects if r['step']==24],curve_moderate_vs_zero=next(x for x in comparisons if x['a']=='curve_lambda1' and x['b']=='curve_lambda0'),continuous_limit=summary['continuous_limit']),indent=2))
if __name__=='__main__':main()
