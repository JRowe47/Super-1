"""No-training resolution sweep, validation-only selection, then held-out tests."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, platform, resource, statistics, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from safetensors.torch import save_file
from representations import install, svd_basis
from prepare_data import model_dir

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';CACHE=ROOT/'cache'
SEED=26317

def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def digest_tensor(t):return hashlib.sha256(memoryview(t.detach().cpu().contiguous().numpy()).cast('B')).hexdigest()

def call_block(model,block,ids):
    x=model.model.embed_tokens(ids)
    pos=torch.arange(ids.shape[1])[None]
    mask=torch.triu(torch.full((ids.shape[1],ids.shape[1]),torch.finfo(x.dtype).min),diagonal=1)[None,None]
    rope=model.model.rotary_emb(x,pos)
    return block(x,attention_mask=mask,position_ids=pos,position_embeddings=rope,use_cache=False)[0]

def calibrate(model,block,windows):
    cov=torch.zeros(576,576,dtype=torch.float64)
    activation_sq=torch.zeros(1536,dtype=torch.float64)
    count=0;gated=[]
    h=block.mlp.down_proj.register_forward_pre_hook(lambda m,a:gated.append(a[0].detach()))
    for row in windows:
        ids=row[:-1][None];x=block.input_layernorm(model.model.embed_tokens(ids)).reshape(-1,576).double()
        cov+=x.T@x;count+=x.shape[0]
        call_block(model,block,ids)
        activation_sq+=gated.pop().double().reshape(-1,1536).square().sum(0)
    h.remove();cov/=count;activation_sq/=count
    ridge=float(cov.diag().mean())*1e-4
    chol=torch.linalg.cholesky(cov+ridge*torch.eye(576,dtype=torch.float64))
    scores=activation_sq*block.mlp.down_proj.weight.double().square().sum(0)
    orders={'importance':torch.argsort(scores,descending=True),'contiguous':torch.arange(1536)}
    for seed in [0,1,2]:orders[f'random_{seed}']=torch.randperm(1536,generator=torch.Generator().manual_seed(seed))
    info={'calibration_tokens':count,'ridge':ridge,'ridge_relative_to_mean_diagonal':1e-4,
          'activation_weighting':'Q/K input second moment after input RMSNorm; uncentered covariance plus ridge.',
          'importance_score':'E[gated_activation_j^2] * ||down_projection_column_j||_2^2; individual contribution energy, omitting cross-neuron correlations.',
          'groups':16,'neurons_per_group':96}
    return chol,orders,info

def get_bases(block,chol):
    bases={'svd':{},'activation_svd':{},'joint_svd':{},'joint_activation_svd':{}};diag=[]
    for parent in [block.self_attn,block.mlp]:
        for name,module in parent.named_children():
            if not hasattr(module,'weight') or module.weight.ndim!=2:continue
            basis=svd_basis(module.weight);bases['svd'][name]=basis
            energy=basis['singular_values'].square().cumsum(0);energy/=energy[-1].clone()
            diag.append({'matrix':name,'shape':list(module.weight.shape),
                         'rank_90_percent_energy':int(torch.searchsorted(energy,torch.tensor(.9,dtype=torch.float64)))+1,
                         'rank_99_percent_energy':int(torch.searchsorted(energy,torch.tensor(.99,dtype=torch.float64)))+1})
    for name in ['q_proj','k_proj']:
        bases['activation_svd'][name]=svd_basis(getattr(block.self_attn,name).weight,chol)
    both=torch.cat([block.self_attn.q_proj.weight,block.self_attn.k_proj.weight])
    bases['joint_svd']['qk']=svd_basis(both)
    bases['joint_activation_svd']['qk']=svd_basis(both,chol)
    return bases,diag

def reference_cache(model,windows,label):
    path=CACHE/f'{label}_reference.pt'
    if path.exists():return torch.load(path,weights_only=True)
    first=[];h=model.model.layers[0].register_forward_hook(lambda m,a,o:first.append(o[0].detach()))
    hs=[];bs=[];nll=[];residual_ss=[]
    for row in windows:
        ids=row[:-1][None];target=row[1:][None]
        hidden=model.model(ids,use_cache=False).last_hidden_state
        logits=model.lm_head(hidden)
        bs.append(first.pop().clone());hs.append(hidden.clone())
        nll.append(float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))))
        residual_ss.append(float((bs[-1]-model.model.embed_tokens(ids)).square().sum()))
    h.remove()
    data={'hidden':torch.cat(hs),'block':torch.cat(bs),'nll':torch.tensor(nll,dtype=torch.float64),
          'residual_ss':torch.tensor(residual_ss,dtype=torch.float64)}
    torch.save(data,path);return data

def evaluate(model,windows,ref):
    first=[];handle=model.model.layers[0].register_forward_hook(lambda m,a,o:first.append(o[0].detach()))
    nlls=[];kls=[];agreements=[];block_ss=0.;max_logit=0.;max_block=0.;finite=True
    started=time.perf_counter()
    for i,row in enumerate(windows):
        ids=row[:-1][None];target=row[1:][None]
        hidden=model.model(ids,use_cache=False).last_hidden_state
        logits=model.lm_head(hidden);gold=model.lm_head(ref['hidden'][i:i+1])
        block=first.pop();delta=block-ref['block'][i:i+1]
        block_ss+=float(delta.square().sum());max_block=max(max_block,float(delta.abs().max()))
        max_logit=max(max_logit,float((logits-gold).abs().max()))
        nlls.append(float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))))
        lp=F.log_softmax(gold,-1);lq=F.log_softmax(logits,-1)
        kls.append(float((lp.exp()*(lp-lq)).sum(-1).mean()))
        agreements.append(float((logits.argmax(-1)==gold.argmax(-1)).float().mean()))
        finite=finite and bool(torch.isfinite(logits).all())
    handle.remove()
    nll=np.asarray(nlls);base=ref['nll'].numpy();delta=nll-base
    return {'windows':len(windows),'predicted_tokens':len(windows)*128,'nll':float(nll.mean()),
            'reference_nll':float(base.mean()),'delta_nll':float(delta.mean()),
            'perplexity_ratio_to_original':float(np.exp(delta.mean())),'mean_kl':float(np.mean(kls)),
            'top1_agreement':float(np.mean(agreements)),
            'block_residual_relative_rms_error':math.sqrt(block_ss/float(ref['residual_ss'].sum())),
            'max_block_abs':max_block,'max_logit_abs':max_logit,'all_finite':finite,
            'per_window_delta_nll':delta.tolist(),'per_window_nll':nll.tolist(),
            'evaluation_seconds':time.perf_counter()-started}

def make_specs():
    specs=[{'id':'exact_curves','method':'exact_curves'},
           {'id':'full_rank_svd_all','method':'svd','target':'all','full_rank':True},
           {'id':'full_rank_activation_qk','method':'activation_svd','target':'qk','full_rank':True},
           {'id':'full_rank_joint_activation_qk','method':'joint_activation_svd','full_rank':True},
           {'id':'full_haar_qk','method':'haar','target':'qk','budget':1.0},
           {'id':'full_knots_qk','method':'knots','target':'qk','budget':1.0},
           {'id':'full_mlp_groups','method':'mlp_groups','groups':16,'ordering':'importance'}]
    for method in ['svd','activation_svd','haar','knots']:
        for target in ['q','k','qk']:
            for b in [.75,.5,.25,.125]:
                specs.append({'id':f'{method}_{target}_{b:g}','method':method,'target':target,'budget':b})
    for method in ['joint_svd','joint_activation_svd']:
        for b in [.75,.5,.25,.125]:
            specs.append({'id':f'{method}_{b:g}','method':method,'budget':b})
    for order in ['importance','contiguous','random_0','random_1','random_2']:
        for groups in [12,8,4,2]:
            specs.append({'id':f'mlp_{order}_{groups}','method':'mlp_groups','groups':groups,'ordering':order})
    specs.append({'id':'mlp_skip','method':'mlp_skip'})
    return specs

def select_for_test(rows):
    # Declared before running the sweep; selection never uses test metrics.
    chosen=[];reason={}
    def choose(pool,label):
        ok=[x for x in pool if x['metrics']['delta_nll']<=.05 and x['metrics']['mean_kl']<=.05]
        if ok:
            item=min(ok,key=lambda x:(x['accounting']['block_projection_macs_per_token'],x['metrics']['mean_kl']))
            why='Lowest projection MAC count passing validation delta NLL <= 0.05 and KL <= 0.05.'
        else:
            item=min(pool,key=lambda x:x['metrics']['mean_kl']);why='No candidate met the gate; chose lowest validation KL as a failed-gate representative.'
        chosen.append(item);reason[item['spec']['id']]=label+': '+why
        return item
    for method in ['svd','activation_svd']:
        choose([r for r in rows if r['spec'].get('method')==method and r['spec'].get('target')=='qk' and not r['spec'].get('full_rank')],method+' Q/K')
    choose([r for r in rows if r['spec']['method']=='joint_activation_svd' and not r['spec'].get('full_rank')],'Joint activation-aware Q/K')
    mlp=choose([r for r in rows if r['spec']['method']=='mlp_groups' and r['spec']['ordering']=='importance' and r['spec']['groups']<16],'Importance-ranked MLP')
    groups=mlp['spec']['groups']
    for id in [f'mlp_random_0_{groups}',f'mlp_contiguous_{groups}','haar_qk_0.5','knots_qk_0.5']:
        item=next(r for r in rows if r['spec']['id']==id);chosen.append(item);reason[id]='Prespecified matched/control comparison.'
    return chosen,reason

def cluster_ci(deltas,docids):
    ids=np.asarray(docids);d=np.asarray(deltas);unique=np.unique(ids)
    sums=np.array([d[ids==i].sum() for i in unique]);counts=np.array([(ids==i).sum() for i in unique])
    idx=np.random.default_rng(SEED+99).integers(0,len(unique),(2000,len(unique)))
    means=sums[idx].sum(1)/counts[idx].sum(1)
    return {'lower':float(np.quantile(means,.025)),'upper':float(np.quantile(means,.975)),
            'clusters':len(unique),'bootstrap_replicates':2000,'unit':'document','quantity':'delta NLL nats/token'}

def timed_pair(model,source,spec,bases,orders,ids):
    original=copy.deepcopy(source)
    install(model,source,spec,bases,orders);converted=model.model.layers[0]
    out={}
    for scope in ['block','full_model']:
        def fn(block):
            if scope=='block':return call_block(model,block,ids)
            model.model.layers[0]=block
            return model(ids,use_cache=False).logits
        for _ in range(3):fn(original);fn(converted)
        pairs=[]
        for i in range(12):
            pair={};sequence=[('original',original),('candidate',converted)]
            if i%2:sequence.reverse()
            for name,block in sequence:
                t=time.perf_counter();fn(block);pair[name]=(time.perf_counter()-t)*1000
            pairs.append(pair)
        out[scope]={'original_median_ms':statistics.median(x['original'] for x in pairs),
                    'candidate_median_ms':statistics.median(x['candidate'] for x in pairs),
                    'median_paired_ratio':statistics.median(x['candidate']/x['original'] for x in pairs),
                    'pairs_ms':pairs}
    model.model.layers[0]=converted
    return out

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['sweep','finish','all'],default='all');args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True);os.environ['TOKENIZERS_PARALLELISM']='false'
    RESULTS.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    start=time.perf_counter()
    data=np.load(ROOT/'data/token_windows.npz')
    windows={k:torch.from_numpy(data[k]).long() for k in data.files}
    manifest=json.loads((ROOT/'data/dataset_manifest.json').read_text())
    model=AutoModelForCausalLM.from_pretrained(model_dir(),local_files_only=True,
              torch_dtype=torch.float32,attn_implementation='eager').eval()
    source=copy.deepcopy(model.model.layers[0])
    outside_hashes={n:digest_tensor(p) for n,p in model.named_parameters() if not n.startswith('model.layers.0.')}
    specs=make_specs()
    protocol={'seed':SEED,'context_tokens':128,'threads':4,'dtype':'float32','attention':'eager',
          'training_steps':0,'scope':'Only block 0; all other parameters unchanged.',
          'budgets':[.75,.5,.25,.125],'budget_definition':'Retained coefficient values divided by the targeted original dense matrix values. SVD ranks account for BOTH factors; full-rank controls are separate.',
          'validation_selection':'For ordinary, activation-aware, joint activation-aware Q/K and importance MLP, choose minimum projection MACs passing delta NLL <= .05 and KL <= .05; otherwise lowest KL representative. Matched random/contiguous MLP, half-budget Haar and knots are prespecified controls.',
          'timing':'12 alternating-order pairs after warmup, 128-token prefill. MAC accounting covers learned linear projections, not attention scores, softmax, normalization or kernels.',
          'specs':specs,'model_manifest':json.loads((ROOT/'model_manifest.json').read_text())}
    dump(RESULTS/'protocol.json',protocol)
    print('Calibrating on',len(windows['train'])*128,'train tokens.',flush=True)
    chol,orders,calibration=calibrate(model,source,windows['train'])
    bases,diag=get_bases(source,chol)
    dump(RESULTS/'calibration.json',calibration);dump(RESULTS/'spectra.json',diag)
    # Orders are required to reproduce the chosen grouped neuron subset.
    dump(RESULTS/'neuron_orders.json',{k:v.tolist() for k,v in orders.items()})
    val=reference_cache(model,windows['validation'],'validation')
    path=RESULTS/'validation_sweep.json'
    rows=json.loads(path.read_text()) if path.exists() else []
    done={r['spec']['id'] for r in rows}
    for j,spec in enumerate(specs):
        if spec['id'] in done:continue
        accounting=install(model,source,spec,bases,orders)
        metrics=evaluate(model,windows['validation'],val)
        row={'spec':spec,'accounting':accounting,'metrics':metrics};rows.append(row);dump(path,rows)
        print(f"[{j+1}/{len(specs)}] {spec['id']} dNLL={metrics['delta_nll']:.6f} KL={metrics['mean_kl']:.6f} agreement={metrics['top1_agreement']:.4f}",flush=True)
        if spec['id']=='exact_curves':assert metrics['max_logit_abs']==0
        if spec.get('full_rank'):assert metrics['max_logit_abs']<1e-3
        assert metrics['all_finite']
    selected,reasons=select_for_test(rows)
    dump(RESULTS/'selection.json',{'selection_used':'validation only','selected':[r['spec'] for r in selected],'reasons':reasons})
    if args.stage=='sweep':
        print('SWEEP COMPLETE',time.perf_counter()-start,flush=True);return
    # Restore reference before generating the held-out cache.
    model.model.layers[0]=copy.deepcopy(source)
    test=reference_cache(model,windows['test'],'test')
    testpath=RESULTS/'heldout_test.json'
    tested=json.loads(testpath.read_text()) if testpath.exists() else []
    done={r['spec']['id'] for r in tested}
    docids=[r['document_index'] for r in manifest['splits']['test']['selected_windows']]
    for row in selected:
        spec=row['spec']
        if spec['id'] in done:continue
        accounting=install(model,source,spec,bases,orders)
        metrics=evaluate(model,windows['test'],test)
        metrics['delta_nll_document_bootstrap_95ci']=cluster_ci(metrics['per_window_delta_nll'],docids)
        tested.append({'spec':spec,'accounting':accounting,'metrics':metrics});dump(testpath,tested)
        print('TEST',spec['id'],'dNLL',round(metrics['delta_nll'],6),'KL',round(metrics['mean_kl'],6),flush=True)
    timing=[]
    for row in selected:
        if row['spec']['method'] not in ['svd','activation_svd','joint_activation_svd','mlp_groups'] or row['spec'].get('ordering','importance')!='importance':continue
        timing.append({'spec':row['spec'],'measurement':timed_pair(model,source,row['spec'],bases,orders,windows['validation'][0,:-1][None])})
        dump(RESULTS/'timing.json',timing)
        print('TIMING',row['spec']['id'],flush=True)
    # Keep complete first-block state for the activation-aware and grouped candidates.
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    for row in selected:
        if row['spec']['method'] not in ['activation_svd','joint_activation_svd','mlp_groups'] or row['spec'].get('ordering','importance')!='importance':continue
        install(model,source,row['spec'],bases,orders)
        state={k:v.detach().contiguous() for k,v in model.model.layers[0].state_dict().items()}
        save_file(state,str(ROOT/'checkpoints'/(row['spec']['id']+'.safetensors')))
    unchanged=all(digest_tensor(dict(model.named_parameters())[n])==h for n,h in outside_hashes.items())
    assert unchanged
    # Cache compatibility and a reproducible generation for the jointly decomposed arm.
    joint=next(r['spec'] for r in selected if r['spec']['method']=='joint_activation_svd')
    install(model,source,joint,bases,orders)
    seq=windows['validation'][0,:32][None]
    full=model(seq,use_cache=False).logits
    cache=DynamicCache();steps=[]
    for i in range(seq.shape[1]):steps.append(model(seq[:,i:i+1],past_key_values=cache,use_cache=True).logits)
    incremental=torch.cat(steps,dim=1);cache_diff=float((full-incremental).abs().max())
    # Different GEMM shapes may change rounding between prefill and decode.
    assert cache_diff<1e-3
    tok=AutoTokenizer.from_pretrained(model_dir(),local_files_only=True)
    prompt='The purpose of a scientific experiment is'
    ids=tok(prompt,return_tensors='pt')['input_ids']
    options={'max_new_tokens':40,'do_sample':False,'pad_token_id':tok.eos_token_id}
    candidate=model.generate(ids,attention_mask=torch.ones_like(ids),**options)
    model.model.layers[0]=copy.deepcopy(source)
    original=model.generate(ids,attention_mask=torch.ones_like(ids),**options)
    summary={'status':'PASS','validation_configs':len(rows),'test_candidates':len(tested),
             'calibration_tokens':len(windows['train'])*128,'validation_tokens':len(windows['validation'])*128,
             'test_tokens':len(windows['test'])*128,'test_reference_nll':float(test['nll'].mean()),
             'untouched_parameter_tensors_checked':len(outside_hashes),'untouched_parameters_identical':unchanged,
             'joint_cached_vs_prefill_max_abs_logit':cache_diff,'joint_cache_check_spec':joint,
             'sample':{'prompt':prompt,'original':tok.decode(original[0,ids.shape[1]:],skip_special_tokens=True),
                       'joint_curve':tok.decode(candidate[0,ids.shape[1]:],skip_special_tokens=True)},
             'run_elapsed_seconds':time.perf_counter()-start,'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             'python':platform.python_version(),'torch':torch.__version__}
    dump(RESULTS/'summary.json',summary)
    print('PHASE 2 COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
