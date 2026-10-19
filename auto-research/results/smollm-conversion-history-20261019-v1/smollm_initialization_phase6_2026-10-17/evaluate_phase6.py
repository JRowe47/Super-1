"""Fresh held-out quality, joint-window ablations and exact curve checks."""
import gc,json,statistics,tempfile,time
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from common import ROOT,OUT,ARMS,dump,setup,load_model,model_dir,data,hashes
from models import candidate,install,restore,lift,install_curve,TIMES,load_candidate
from evaluation import evaluate,cluster_ci
from diagnostics import curve_profiles

def configurations():
    return [
        {'id':'initial_interpolated','kind':'interpolated','trained':False},
        {'id':'initial_retained','kind':'retained','trained':False},
        {'id':'phase5_dense','kind':'phase5','trained':False},
        {'id':'trained_interpolated','kind':'interpolated','trained':True},
        {'id':'trained_retained','kind':'retained','trained':True}]

def make_blocks(original,config):
    if config['kind']=='phase5':
        return candidate(original,'retained',checkpoint=ROOT/'reference/phase5_dense_n3.safetensors')
    checkpoint=ROOT/'checkpoints'/f'{config["kind"]}.safetensors' if config['trained'] else None
    return candidate(original,config['kind'],checkpoint=checkpoint)

@torch.no_grad()
def evaluate_split(model,original,rows,windows,reference,docids,path):
    completed=json.loads(path.read_text()) if path.exists() else [];done={r['configuration']['id'] for r in completed}
    for config in rows:
        if config['id'] in done:continue
        blocks=make_blocks(original,config);install(model,original,blocks,only=config.get('only'))
        m=evaluate(model,windows,reference);assert m['all_finite']
        m['delta_nll_document_bootstrap_95']=cluster_ci(m['per_window_delta_nll'],docids)
        completed.append({'configuration':config,'decoder_blocks':len(model.model.layers),'metrics':m});dump(path,completed)
        print(path.stem.upper(),config['id'],'PPL',round(m['perplexity_ratio_to_original'],6),'KL',round(m['mean_kl'],6),flush=True)
        restore(model,original);del blocks;gc.collect()
    return completed

@torch.no_grad()
def cache_check(model,ids):
    expected=model(ids,use_cache=False).logits;cache=DynamicCache();rows=[]
    for i in range(ids.shape[1]):rows.append(model(ids[:,i:i+1],past_key_values=cache,use_cache=True).logits)
    difference=float((expected-torch.cat(rows,1)).abs().max())
    return {'max_cached_vs_prefill_logit_difference':difference,'threshold':.001,'passed':difference<.001,
        'kv_layers':len(cache),'cached_tokens':ids.shape[1],
        'kv_bytes':sum(t.numel()*t.element_size() for t in cache.key_cache+cache.value_cache)}

@torch.no_grad()
def time_candidate(model,original,blocks,banks,ids):
    result={}
    for mode in ['native_blocks','streamed_curve']:
        if mode=='native_blocks':install(model,original,blocks)
        else:install_curve(model,original,banks,3)
        layers=list(model.model.layers);pairs=[]
        for repeat in range(14):
            values={}
            for label in (['original','candidate'] if repeat%2==0 else ['candidate','original']):
                if label=='original':restore(model,original)
                else:
                    model.model.layers=torch.nn.ModuleList(layers);model.config.num_hidden_layers=len(layers)
                    for i,block in enumerate(layers):
                        if hasattr(block,'self_attn'):block.self_attn.layer_idx=i
                start=time.perf_counter();model(ids,use_cache=False).logits
                values[label]=(time.perf_counter()-start)*1000
            if repeat>=2:pairs.append(values)
        result[mode]={'original_median_ms':statistics.median(p['original'] for p in pairs),
            'candidate_median_ms':statistics.median(p['candidate'] for p in pairs),
            'median_paired_ratio':statistics.median(p['candidate']/p['original'] for p in pairs),
            'pairs_ms':pairs,'warmup_pairs':2,'recorded_pairs':12,'context':ids.shape[1],
            'batch':ids.shape[0],'threads':4,'precision':'FP32','scope':'CPU prefill, excludes curve fitting and native model construction'}
        restore(model,original)
    return result

@torch.no_grad()
def verification(model,original,windows):
    baseline_hashes=hashes(model);ids=windows['audit'][:1,:-1]
    original_logits=model(ids,use_cache=False).logits;result={};timing={}
    profile_banks={}
    for kind in ARMS:
        blocks=candidate(original,kind,checkpoint=ROOT/'checkpoints'/f'{kind}.safetensors')
        banks=lift(original,blocks);profile_banks['trained_'+kind]=banks
        maximum_weight=0.;maximum_logit=0.
        for s in ['4','12']:
            for i,t in enumerate(TIMES):
                expected=dict(blocks[s][i].named_parameters());actual=banks[s].sample(t)
                maximum_weight=max(maximum_weight,max(float((expected[name]-actual[name]).abs().max()) for name in expected))
        for start in range(0,8,2):
            batch=windows['audit'][start:start+2,:-1]
            install(model,original,blocks);expected=model(batch,use_cache=False).logits
            install_curve(model,original,banks,3);actual=model(batch,use_cache=False).logits
            maximum_logit=max(maximum_logit,float((expected-actual).abs().max()))
        curve_cache=cache_check(model,ids[:,:32])
        install_curve(model,original,banks,4)
        four_difference=float((model(ids,use_cache=False).logits-original_logits).abs().max())
        assert four_difference==0
        install(model,original,blocks);native_cache=cache_check(model,ids[:,:32])
        fresh=candidate(original,kind,checkpoint=ROOT/'checkpoints'/f'{kind}.safetensors')
        expected=model(ids[:,:32],use_cache=False).logits
        install(model,original,fresh);reloaded=model(ids[:,:32],use_cache=False).logits
        reload_difference=float((expected-reloaded).abs().max());assert reload_difference==0
        native_parameter_count=sum(p.numel() for p in model.parameters())
        result[kind]={'all_sampled_weight_max_abs_difference':maximum_weight,
            'curve_vs_native_max_logit_difference':maximum_logit,'logit_equivalence_tokens':1024,
            'four_step_original_max_logit_difference':four_difference,
            'checkpoint_reload_max_logit_difference':reload_difference,
            'native_parameter_count':native_parameter_count,'curve_cache':curve_cache,'native_cache':native_cache,
            'passed':maximum_weight<1e-6 and maximum_logit<.001 and curve_cache['passed'] and native_cache['passed']}
        assert result[kind]['passed'],result[kind]
        timing[kind]=time_candidate(model,original,blocks,banks,ids)
        restore(model,original);del blocks,fresh;gc.collect()
        print('CURVE_CHECK',kind,result[kind],flush=True)
    for kind in ARMS:
        blocks=candidate(original,kind);profile_banks['initial_'+kind]=lift(original,blocks);del blocks
    curve_profiles(profile_banks)
    restore(model,original);assert baseline_hashes==hashes(model)
    result['original_parameter_hashes_unchanged']=True
    dump(OUT/'checks.json',result);dump(OUT/'timing.json',timing)

@torch.no_grad()
def main():
    setup();windows=data();manifest=json.loads((ROOT/'data/dataset_manifest.json').read_text())
    model=load_model();original=list(model.model.layers);before=hashes(model)
    reference=torch.load(ROOT/'cache/validation.pt',weights_only=True)
    rows=configurations()
    val=evaluate_split(model,original,rows,windows['validation'],reference,
        [r['document_index'] for r in manifest['validation_windows_manifest']],OUT/'validation.json')
    previous=json.loads((ROOT/'reference/phase5_validation.json').read_text())
    expected=next(r['metrics'] for r in previous if r['configuration']['id']=='dense_n3')
    actual=next(r['metrics'] for r in val if r['configuration']['id']=='phase5_dense')
    reference_errors={k:abs(actual[k]-expected[k]) for k in ['delta_nll','mean_kl','top1_agreement']}
    # Argmax is discontinuous near ties; use a separate agreement tolerance
    # from the continuous NLL/KL metrics when residual scaling is folded.
    reference_tolerances={'delta_nll':1e-5,'mean_kl':1e-5,'top1_agreement':.001}
    assert all(v<reference_tolerances[k] for k,v in reference_errors.items()),reference_errors
    dump(OUT/'phase5_reference_check.json',{'same_validation_windows':True,'absolute_metric_differences':reference_errors,
        'tolerances':reference_tolerances,'passed':True,'meaning':'Compiled reference reproduces the selected phase-5 dense validation metrics; folding changes FP32 roundoff, which can affect argmax near ties.'})
    passed={r['configuration']['id']:r['metrics']['delta_nll']<=.05 and r['metrics']['mean_kl']<=.05 for r in val}
    dump(OUT/'selection.json',{'selected_steps':{k:json.loads((OUT/f'training_{k}.json').read_text())['best_update'] for k in ARMS},
        'passed_gate':passed,'gate':'Full 32 validation windows: delta NLL <= .05 AND teacher KL <= .05.',
        'no_audit_based_selection':True,'primary_comparison':'trained_retained versus trained_interpolated',
        'checkpoint_rule':'Minimum teacher KL on first 8 validation windows, checked every 300 updates including 0.'})
    reference=torch.load(ROOT/'cache/audit.pt',weights_only=True)
    docids=[r['document_index'] for r in manifest['audit_windows_manifest']]
    for kind in ARMS:
        for start in [4,12]:rows.append({'id':f'trained_{kind}_only{start}','kind':kind,'trained':True,'only':[start]})
    evaluate_split(model,original,rows,windows['audit'],reference,docids,OUT/'audit.json')
    long=[r for r in configurations() if r['trained'] or r['kind']=='phase5']
    reference=torch.load(ROOT/'cache/audit_long.pt',weights_only=True)
    evaluate_split(model,original,long,windows['audit_long'],reference,docids,OUT/'long_context.json')
    verification(model,original,windows)
    tokenizer=AutoTokenizer.from_pretrained(model_dir(),local_files_only=True);samples=[]
    for label in ['original',*ARMS]:
        if label=='original':restore(model,original)
        else:
            blocks=candidate(original,label,checkpoint=ROOT/'checkpoints'/f'{label}.safetensors');install(model,original,blocks)
        for prompt in ['The purpose of a computer network is','A useful way to test a scientific theory is']:
            ids=tokenizer(prompt,return_tensors='pt').input_ids
            tokens=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=40,
                do_sample=False,pad_token_id=tokenizer.eos_token_id,use_cache=True)
            samples.append({'model':label,'prompt':prompt,
                'new_token_count':int(tokens.shape[1]-ids.shape[1]),
                'continuation':tokenizer.decode(tokens[0,ids.shape[1]:],skip_special_tokens=True)})
    dump(OUT/'generations.json',samples)
    restore(model,original);assert before==hashes(model)
    print('EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
