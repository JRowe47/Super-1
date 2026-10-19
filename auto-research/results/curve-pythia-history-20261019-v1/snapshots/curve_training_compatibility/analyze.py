"""Paired continuation statistics, conversion diagnostics, and finite-task limits."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def read(n):return json.loads((OUT/n).read_text())
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def ci(v,indices,weights=None):
    v=np.asarray(v,np.float64)
    if weights is None:est=v.mean();s=v[indices].mean(1)
    else:
        w=np.asarray(weights);est=np.dot(v,w)/w.sum();s=(v[indices]*w[indices]).sum(1)/w[indices].sum(1)
    return dict(estimate=float(est),ci95=np.quantile(s,[.025,.975]).tolist())
def seq_indices(rows):
    rng=np.random.default_rng(2026101711);groups=[]
    for n in sorted({r['n'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['n']==n];groups.append(rng.choice(ids,(4000,len(ids)),replace=True))
    return np.concatenate(groups,1)
def effects(rows,group):
    result=[]
    for arm in sorted({r['arm'] for r in rows}):
        base=next(r for r in rows if r['arm']==arm and r['step']==0)
        for row in [r for r in rows if r['arm']==arm]:
            idx=np.random.default_rng(2026101712).integers(0,24,(4000,24))
            e=dict(group=group,arm=arm,step=row['step'],validation_nll=row['validation']['nll'],
                validation_delta=ci(np.array(row['validation']['block_nll'])-base['validation']['block_nll'],idx))
            for kind in ['repeat','scrambled']:
                b=base['retention'][kind];r=row['retention'][kind];ix=seq_indices(b['sequences'])
                e[kind]=dict(accuracy=r['accuracy'],nll=r['target_nll'])
                for key in ['target_nll','accuracy']:
                    vals=np.array([a[key]-c[key] for a,c in zip(r['sequences'],b['sequences'])]);vals*=100 if key=='accuracy' else 1
                    e[kind][key+'_delta']=ci(vals,ix)
            b=base['retention']['prose'];r=row['retention']['prose'];ix=np.random.default_rng(2026101713).integers(0,8,(4000,8))
            e['prose']=dict(nll=r['nll'],nll_delta=ci([a['nll']-c['nll'] for a,c in zip(r['passages'],b['passages'])],ix,[a['tokens'] for a in r['passages']]),
                kl=r['kl_from_baseline'],argmax_changed=r['argmax_changed'],tokens=r['tokens'])
            result.append(e)
    return result
def main():
    rows=read('pythia_evaluations.json');stable=read('stable_sgd_evaluations.json');eff=effects(rows,'original')+effects(stable,'stable_sgd');dump('pythia_effects.json',eff)
    a=next(r for r in rows if r['arm']=='curve_adamw' and r['step']==24);b=next(r for r in rows if r['arm']=='native_adamw' and r['step']==24)
    ix=np.random.default_rng(2026101712).integers(0,24,(4000,24));contrasts=dict(curve_minus_native_validation=ci(np.array(a['validation']['block_nll'])-b['validation']['block_nll'],ix))
    for kind in ['repeat','scrambled']:
        ar=a['retention'][kind]['sequences'];br=b['retention'][kind]['sequences'];sx=seq_indices(br)
        contrasts['curve_minus_native_'+kind+'_accuracy_pp']=ci([100*(r['accuracy']-s['accuracy']) for r,s in zip(ar,br)],sx)
    dump('pythia_paired_contrasts.json',contrasts)
    trajectories=read('toy_trajectories.json');arms=sorted({r['arm'] for r in trajectories});toy=[]
    for arm in arms:
        for step in sorted({r['step'] for r in trajectories if r['arm']==arm}):
            runs=sorted([r for r in trajectories if r['arm']==arm and r['step']==step],key=lambda r:r['seed']);assert len(runs)==5
            for depth in [3,4,6,8,12,24]:
                ms=[next(m for m in r['metrics'] if m['depth']==depth) for r in runs]
                out=dict(arm=arm,step=step,depth=depth,active_parameters=runs[0]['active_parameters'],seeds=[r['seed'] for r in runs])
                for key in ['bce','accuracy','joint_accuracy','logit_rms_from_depth6']:
                    vals=[m[key] for m in ms];out[key]=dict(mean=float(np.mean(vals)),min=float(min(vals)),max=float(max(vals)),values=vals)
                toy.append(out)
    dump('toy_summary.json',toy)
    # Post-hoc diagnostic: determine whether the finite tasks actually use the residual stages.
    inp=np.array(list(__import__('itertools').product([-1.,1.],repeat=4)));aa,bb,cc,dd=inp.T
    target=np.stack([aa,aa*bb,np.where(cc<0,aa,bb),aa*bb*cc*dd,np.sign(aa+bb+cc),bb*cc*dd],1)
    ablations=[]
    for arm in arms:
        step=400 if arm in ['curve6_exact','curve2_hard','curve2_bridge'] else 1200
        for seed in range(5):
            with np.load(ROOT/'models'/f'toy_seed{seed}_{arm}_{step}.npz') as z:logits=np.tanh(inp@z['A'].T+z['ai'])@z['O'].T+z['oi']
            ok=(logits>0)==(target>0)
            ablations.append(dict(arm=arm,seed=seed,accuracy=float(ok.mean()),joint_accuracy=float(ok.all(1).mean()),bce=float(np.logaddexp(0,-target*logits).mean())))
    dump('toy_posthoc_zero_residual.json',dict(description='Post-hoc no-training control prompted by ceiling accuracy. Uses trained input/output maps with every residual stage omitted; never used to select models.',rows=ablations))
    eq=read('pythia_equivalence.json');seq=read('stable_sgd_equivalence.json')
    summary=dict(pythia_final=[r for r in eff if r['step']==24],paired_contrasts=contrasts,
        adam_transport_equivalence=[r for r in eq if r['arm']=='curve_transport_adamw'],stable_sgd_equivalence=seq,
        oversized_sgd_equivalence=[r for r in eq if r['arm']=='curve_sgd'],first_step=read('pythia_first_step_checks.json'),
        toy_final=[r for r in toy if r['step']==(400 if r['arm'] in ['curve6_exact','curve2_hard','curve2_bridge'] else 1200)],
        toy_conversion=read('toy_conversion_diagnostics.json'),toy_zero_residual=ablations,
        runtime=dict(pythia=read('pythia_environment.json')['seconds'],stable_sgd=read('stable_sgd_environment.json')['seconds'],toy=read('toy_environment.json')['seconds']),
        scope='19M decoder parameters continued for 24 batches per arm; no new pretraining. Five-seed finite truth-table toy is a controlled depth field, not the full GSA architecture.')
    dump('summary.json',summary)
    print(json.dumps(dict(pythia_final=summary['pythia_final'],contrasts=contrasts,stable_sgd=seq,adam_transport=summary['adam_transport_equivalence'],
        update_comparisons=summary['first_step']['update_comparisons'],toy_zero_residual=ablations,runtime=summary['runtime']),indent=2))
if __name__=='__main__':main()
