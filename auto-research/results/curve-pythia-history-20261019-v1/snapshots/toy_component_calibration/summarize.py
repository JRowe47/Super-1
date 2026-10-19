"""Summarize the frozen experiment; no fitting or hyperparameter selection."""
import json
from pathlib import Path
import numpy as np
from scipy import fft
import torch
from calibrate import ROOT,OUT,MODELS,STEPS,INTERVALS,X,WALSH,haar,ihaar,infer,projections,Toy,dump,rms

def load(name): return json.loads((OUT/name).read_text())

def main():
    runs=load('runs.json'); ls=load('leaf_summaries.json'); si=load('sibling_interactions.json')
    functional=[]; selectivity=[]
    for r in runs:
        z=np.load(MODELS/f"{r['task']}_seed{r['seed']}.npz")
        for ti,step in enumerate(STEPS):
            pars=[z[f'p{k}'][ti] for k in range(6)]
            hf=[];ff=[]
            for p in pars:
                c,l=haar(p);hf.append(ihaar(c,l).reshape(p.shape))
                ff.append(fft.irfft(fft.rfft(p.ravel(),norm='ortho'),n=p.size,norm='ortho').reshape(p.shape))
            actual=z['original_logits'][ti]
            functional.append(dict(task=r['task'],seed=r['seed'],step=step,
                haar_logit_max_abs=float(np.max(abs(infer(hf)-actual))),fourier_logit_max_abs=float(np.max(abs(infer(ff)-actual))),
                haar_same_choices=bool(np.array_equal(infer(hf)>0,actual>0)),fourier_same_choices=bool(np.array_equal(infer(ff)>0,actual>0))))
        if r['task']=='joint':
            cat=json.loads(str(z['catalog_json'])); logits=z['rollback_logits'][-1];targets=z['targets']
            acc=np.mean((logits>0)==(targets>0)[None],axis=1)
            q=np.tanh(logits/2);orig=np.tanh(z['original_logits'][-1]/2)
            drops=np.mean(orig*targets,axis=0)[None]-np.mean(q*targets[None],axis=1)
            ids=[i for i,c in enumerate(cat) if c['basis']=='haar' and c['tid']<4]
            selected=[i for i in ids if acc[i,2]<1 and acc[i,0]==1 and acc[i,1]==1]
            best=max(ids,key=lambda i:drops[i,2]-np.max(abs(drops[i,:2])))
            selectivity.append(dict(seed=r['seed'],hidden_haar_atoms=len(ids),decision_selective_context_atoms=len(selected),
                best_margin_selectivity_component=best,accuracy=acc[best].tolist(),target_margin_drop=drops[best].tolist(),
                best_component=cat[best],decision_selective_component_ids=selected))
    dump('functional_curve_checks.json',functional);dump('joint_selectivity.json',selectivity)
    # A labelled illustrative example, selected after the exhaustive scan.
    z=np.load(MODELS/'joint_seed1.npz');cat=json.loads(str(z['catalog_json']));i=197;ti=2;atom=37
    assert cat[i]['tensor']=='l1.weight' and cat[i]['basis']=='haar' and cat[i]['atom']==atom
    old=[z[f'p{k}'][0] for k in range(6)];new=[z[f'p{k}'][-1] for k in range(6)]
    delta=[b-a for a,b in zip(old,new)];component=projections(delta,cat)[i]
    q=np.tanh(z['original_logits'][-1]/2);rlog=z['rollback_logits'][-1,i];qr=np.tanh(rlog/2)
    dose=[]
    for strength in [0,.25,.5,.75,1]:
        pp=[p.copy() for p in new];pp[ti]-=strength*component;l=infer(pp)
        dose.append(dict(strength=strength,accuracy=np.mean((l>0)==(z['targets']>0),axis=0).tolist(),
            target_margin_drop=(np.mean(q*z['targets'],axis=0)-np.mean(np.tanh(l/2)*z['targets'],axis=0)).tolist()))
    pp=[p.copy() for p in new];pp[ti]-=component
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);m=Toy(3)
    with torch.no_grad():
        for par,v in zip(m.parameters(),pp):par.copy_(torch.tensor(v))
    replay=m(torch.tensor(X)).detach().numpy()
    response=np.tanh(z['transplant_logits'][-1,i]/2)-np.tanh(z['original_logits'][0]/2)
    example=dict(selection='Post-hoc illustration: joint seed 1, strongest hidden Haar atom by context target-margin drop minus largest absolute off-target drop in that seed; seed-specific, not a transferable address.',
        seed=1,component=i,metadata=cat[i],interval=[0,800],before_coefficient=float(z['haar2'][0,atom]),after_coefficient=float(z['haar2'][-1,atom]),
        coefficient_trajectory=z['haar2'][:,atom].tolist(),native_parameter_delta=component.ravel().tolist(),
        affected_parameters=[dict(index=[int(v) for v in np.unravel_index(k,component.shape)],delta=float(component.ravel()[k])) for k in np.flatnonzero(abs(component.ravel())>1e-15)],
        baseline_accuracy=np.mean((z['original_logits'][-1]>0)==(z['targets']>0),axis=0).tolist(),
        rollback_accuracy=np.mean((rlog>0)==(z['targets']>0),axis=0).tolist(),
        target_margin_drop=(np.mean(q*z['targets'],axis=0)-np.mean(qr*z['targets'],axis=0)).tolist(),
        function_rms_effect=rms(q-qr,axis=0).tolist(),
        failed_context_inputs=X[(rlog[:,2]>0)!=(z['targets'][:,2]>0)].tolist(),
        branch_accuracy={str(v):float(np.mean((rlog[X[:,2]==v,2]>0)==(z['targets'][X[:,2]==v,2]>0))) for v in [-1,1]},
        function_feature_changes=((q-qr).T@WALSH/16).tolist(),
        baseline_initial_accuracy=np.mean((z['original_logits'][0]>0)==(z['targets']>0),axis=0).tolist(),
        transplant_accuracy=np.mean((z['transplant_logits'][-1,i]>0)==(z['targets']>0),axis=0).tolist(),
        transplant_function_rms=rms(response,axis=0).tolist(),dose=dose,torch_replay_max_abs=float(np.max(abs(replay-rlog))))
    dump('illustrative_component.json',example)
    late=[r for r in ls if r['start']==0 and r['end']==800 and r['basis']=='haar']
    corrs=np.array([r['change_effect_spearman'] for r in late])
    inter={}
    for name in ['l0.weight','l1.weight','out.weight']:
        rows=[r for r in si if r['start']==0 and r['end']==800 and r['tensor']==name and r['child_logit_effect_sum']>1e-8]
        v=np.array([r['logit_relative'] for r in rows])
        inter[name]=dict(count=len(rows),median_relative=float(np.median(v)),p90_relative=float(np.quantile(v,.9)),max_relative=float(max(v)))
    # Keep per-model medians to show seed variability, rather than treat all nodes as independent replications.
    inter_runs=[]
    for rr in runs:
        for name in ['l0.weight','l1.weight','out.weight']:
            rows=[r for r in si if r['task']==rr['task'] and r['seed']==rr['seed'] and r['start']==0 and r['end']==800 and r['tensor']==name and r['child_logit_effect_sum']>1e-8]
            inter_runs.append(dict(task=rr['task'],seed=rr['seed'],tensor=name,median_relative=float(np.median([r['logit_relative'] for r in rows]))))
    dump('interaction_run_medians.json',inter_runs)
    checks=load('numerical_checks.json')
    summary=dict(runs=len(runs),all_final_truth_tables_correct=all(all(a==1 for a in r['final_accuracy']) for r in runs),
        functional_reconstruction_max_abs=max(max(r['haar_logit_max_abs'],r['fourier_logit_max_abs']) for r in functional),
        all_reconstructed_choices_identical=all(r['haar_same_choices'] and r['fourier_same_choices'] for r in functional),
        parameter_tree_additivity_max_abs=max(r['parameter_additivity_max_abs'] for r in si),
        haar_change_effect_spearman=dict(median=float(np.median(corrs)),minimum=float(min(corrs)),maximum=float(max(corrs))),
        sibling_interaction_by_tensor=inter,
        joint_seeds_with_decision_selective_hidden_context_component=sum(r['decision_selective_context_atoms']>0 for r in selectivity),
        seeds=5,example=example,environment=load('environment.json'),
        caveats=['All 16 inputs are both trained and enumerated; no held-out generalization or grokking claim.',
            'Single component rollbacks and sibling comparisons are exhaustive only within the specified hierarchy.',
            'Natural parameter-change size is not equalized; effect-size correlations are descriptive and not a basis-comparison result.',
            'Probability saturation hides some changes from accuracy; logit and probability effects are stored separately.',
            'The example was selected after scanning; effects in 3/5 seeds do not establish universality or an address that transfers between models.',
            'Haar support is in the tensor index space. Boolean function features are a separate space of known input meanings.',
            'Our toy tasks do not reproduce Toy Models of Superposition and do not measure superposition directly.'])
    dump('summary.json',summary)
    assert summary['all_reconstructed_choices_identical']
    assert summary['functional_reconstruction_max_abs']<1e-12
    assert example['torch_replay_max_abs']<1e-12
    assert summary['parameter_tree_additivity_max_abs']<1e-12
    print(json.dumps({k:v for k,v in summary.items() if k not in ['example','caveats']},indent=2))

if __name__=='__main__':main()
