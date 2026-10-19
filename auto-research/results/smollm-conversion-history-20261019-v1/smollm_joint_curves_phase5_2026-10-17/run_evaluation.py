import gc,json,math,resource,statistics,time
import numpy as np
import torch
from transformers import AutoTokenizer,DynamicCache
from common import ROOT,OUT,ARMS,dump,setup,load_model,model_dir,hashes,data
from curve_model import restore
from joint_model import make_banks,set_model,load_banks,project_dense,lift_dense,export_native,TIMES
from prepare import cache_split
from evaluation import evaluate,cluster_ci
from diagnostics import profiles


def configs():
    return [{'id':f'{kind}_n{n}','kind':kind,'steps':n,'only':None}
            for kind in ['initial',*ARMS,'projected_dense','lifted_dense'] for n in [3,2]]


def configure(model,original,banks,config):
    set_model(model,original,banks[config['kind']],config['steps'],only=config.get('only'))


@torch.no_grad()
def cached_check(model,ids):
    full=model(ids,use_cache=False).logits;cache=DynamicCache();rows=[]
    for i in range(ids.shape[1]):rows.append(model(ids[:,i:i+1],past_key_values=cache,use_cache=True).logits)
    diff=float((full-torch.cat(rows,1)).abs().max())
    return {'max_cached_vs_prefill_logit_abs_difference':diff,'passed':diff<.001,'threshold':.001,
        'kv_layers':len(cache),'kv_bytes':sum(t.numel()*t.element_size() for t in cache.key_cache+cache.value_cache),
        'cached_tokens':ids.shape[1]}


@torch.no_grad()
def timing(model,original,bank,n,ids):
    result={}
    for mode in ['streamed','native_export']:
        # Keep the constructed layer list, so setup/copying is outside timing.
        if mode=='streamed':set_model(model,original,bank,n)
        else:export_native(model,original,bank,n)
        candidate=list(model.model.layers);pairs=[]
        for repeat in range(14):
            row={}
            for which in (['native','candidate'] if repeat%2==0 else ['candidate','native']):
                if which=='native':restore(model,original)
                else:
                    model.model.layers=torch.nn.ModuleList(candidate)
                    for j,b in enumerate(candidate):
                        if hasattr(b,'self_attn'):b.self_attn.layer_idx=j
                begin=time.perf_counter();model(ids,use_cache=False).logits
                row[which]=(time.perf_counter()-begin)*1000
            if repeat>=2:pairs.append(row)
        result[mode]={'native_median_ms':statistics.median(p['native'] for p in pairs),
            'candidate_median_ms':statistics.median(p['candidate'] for p in pairs),
            'median_paired_ratio':statistics.median(p['candidate']/p['native'] for p in pairs),
            'pairs_ms':pairs,'context':ids.shape[1],'batch':ids.shape[0],'threads':4,
            'note':'12 alternating paired prefill timings; 2 warmup pairs. Setup/sampling into native blocks is excluded for native_export; streamed forward includes function-weight sampling.'}
        restore(model,original)
    return result


@torch.no_grad()
def evaluate_split(model,original,banks,rows,windows,reference,docids,path):
    done_rows=json.loads(path.read_text()) if path.exists() else [];done={r['configuration']['id'] for r in done_rows}
    for config in rows:
        if config['id'] in done:continue
        configure(model,original,banks,config);m=evaluate(model,windows,reference);assert m['all_finite']
        m['delta_nll_document_bootstrap_95']=cluster_ci(m['per_window_delta_nll'],docids)
        count=len(model.model.layers)
        done_rows.append({'configuration':config,'decoder_blocks':count,'metrics':m});dump(path,done_rows)
        print(path.stem.upper(),config['id'],'PPL',round(m['perplexity_ratio_to_original'],5),'KL',round(m['mean_kl'],5),flush=True)
    return done_rows


@torch.no_grad()
def main():
    setup();windows=data();manifest=json.loads((ROOT/'data/dataset_manifest.json').read_text())
    model=load_model();original=list(model.model.layers);before=hashes(model);banks={}
    banks['initial']=make_banks(original,'raw_curve')
    for kind in ARMS:
        banks[kind]=make_banks(original,kind);load_banks(banks[kind],ROOT/'checkpoints'/f'{kind}.safetensors')
    banks['projected_dense'],projection=project_dense(original,banks['dense']);dump(OUT/'projection.json',projection)
    banks['lifted_dense'],lifting=lift_dense(original,banks['dense']);dump(OUT/'lifting.json',lifting)
    # No optimized checkpoint is selected using the fresh audit.
    validation_ref=torch.load(ROOT/'cache/validation.pt',weights_only=True)
    rows=configs();val=evaluate_split(model,original,banks,rows,windows['validation'],validation_ref,
        [r['document_index'] for r in manifest['validation_windows_manifest']],OUT/'validation.json')
    gates={r['configuration']['id']:r['metrics']['delta_nll']<=.05 and r['metrics']['mean_kl']<=.05 for r in val}
    dump(OUT/'selection.json',{'selection_split':'first 8 validation windows for checkpoints; all 32 for gate reporting',
        'passed_gate':gates,'audit_rows_locked':rows,'primary_curve':'orthogonal_curve',
        'no_test_based_resolution_selection':True,'extra_ablations':'orthogonal_curve and dense, N=3, each individual window',
        'long_context':['initial_n3','orthogonal_curve_n3','dense_n3']})
    restore(model,original);audit_ref=cache_split(model,windows['audit'],'audit')
    for kind in ['orthogonal_curve','dense']:
        for start in [4,12]:rows.append({'id':f'{kind}_n3_only{start}','kind':kind,'steps':3,'only':[start]})
    docids=[r['document_index'] for r in manifest['audit_windows_manifest']]
    audit=evaluate_split(model,original,banks,rows,windows['audit'],audit_ref,docids,OUT/'audit.json')
    longrows=[c for c in configs() if c['id'] in ['initial_n3','orthogonal_curve_n3','dense_n3']]
    restore(model,original);longref=cache_split(model,windows['audit_long'],'audit_long')
    evaluate_split(model,original,banks,longrows,windows['audit_long'],longref,docids,OUT/'long_context.json')
    integrity={};cachechecks={};timings={}
    for kind in ['orthogonal_curve','dense']:
        for n in [3,2]:
            set_model(model,original,banks[kind],n);ids=windows['audit'][:1,:-1]
            curve=model(ids,use_cache=False).logits;cachechecks[f'{kind}_n{n}']=cached_check(model,ids[:,:32])
            export_native(model,original,banks[kind],n);native=model(ids,use_cache=False).logits
            difference=float((curve-native).abs().max())
            integrity[f'{kind}_n{n}']={'native_export_max_logit_difference':difference,'passed':difference<.001,
                'native_export_unique_parameter_values':sum(p.numel() for p in model.parameters()),
                'native_export_cache':cached_check(model,ids[:,:32])}
            print('EXPORT',kind,n,integrity[f'{kind}_n{n}'],flush=True)
            timings[f'{kind}_n{n}']=timing(model,original,banks[kind],n,windows['audit'][:1,:-1])
    # Every curve still lands on original immutable knots after joint training.
    restore(model,original);reference=model(windows['audit'][:1,:-1],use_cache=False).logits
    exact={}
    for kind in ['raw_curve','orthogonal_curve','projected_dense','lifted_dense']:
        set_model(model,original,banks[kind],4);out=model(windows['audit'][:1,:-1],use_cache=False).logits
        exact[kind]=float((out-reference).abs().max());assert exact[kind]==0
    # Reload the selected orthogonal bank, independently reconstructing bases.
    fresh=make_banks(original,'orthogonal_curve');load_banks(fresh,ROOT/'checkpoints/orthogonal_curve.safetensors')
    set_model(model,original,fresh,3);a=model(windows['audit'][:1,:32],use_cache=False).logits
    set_model(model,original,banks['orthogonal_curve'],3);b=model(windows['audit'][:1,:32],use_cache=False).logits
    reload_diff=float((a-b).abs().max());assert reload_diff==0
    lift_checks={}
    for n in [3,2]:
        maxdiff=0.;weightdiff=0.
        for s in ['4','12']:
            for t in (TIMES[:3] if n==3 else TIMES[3:]):
                dense_weights=banks['dense'][s].sample(t);lifted_weights=banks['lifted_dense'][s].sample(t)
                weightdiff=max(weightdiff,max(float((dense_weights[name]-lifted_weights[name]).abs().max()) for name in dense_weights))
        for start in range(0,8,2):
            ids=windows['audit'][start:start+2,:-1]
            set_model(model,original,banks['dense'],n);a=model(ids,use_cache=False).logits
            set_model(model,original,banks['lifted_dense'],n);b=model(ids,use_cache=False).logits
            maxdiff=max(maxdiff,float((a-b).abs().max()))
        lift_checks[str(n)]={'dense_vs_lifted_max_logit_difference':maxdiff,'all_sampled_weight_max_abs_difference':weightdiff,
            'threshold':.001,'passed':maxdiff<.001 and weightdiff<1e-6,'audit_tokens':1024,
            'coverage':'All sampled tensors in both windows; end-to-end logits on first 8 audit windows. Separate quality metrics cover all 64.'}
    restore(model,original);assert before==hashes(model)
    dump(OUT/'checks.json',{'native_exports':integrity,'curve_cache_checks':cachechecks,'exact_four_step_logit_differences':exact,
        'reloaded_orthogonal_curve_max_logit_difference':reload_diff,'dense_to_curve_lift':lift_checks,'original_parameter_hashes_unchanged':True})
    dump(OUT/'timing.json',timings)
    profiles(banks)
    # Fixed prompts: illustration only, not a quality score or selection input.
    tok=AutoTokenizer.from_pretrained(model_dir(),local_files_only=True)
    prompts=['The purpose of a computer network is','A useful way to test a scientific theory is']
    samples=[]
    for label in ['original','orthogonal_curve','dense']:
        if label=='original':restore(model,original)
        else:export_native(model,original,banks[label],3)
        for prompt in prompts:
            ids=tok(prompt,return_tensors='pt').input_ids
            output=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=40,
                do_sample=False,pad_token_id=tok.eos_token_id,use_cache=True)
            samples.append({'model':label,'prompt':prompt,'continuation':tok.decode(output[0,ids.shape[1]:],skip_special_tokens=True)})
    dump(OUT/'generations.json',samples);restore(model,original)
    print('EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
