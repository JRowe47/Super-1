"""Post-hoc sparse update interventions; no optimization of model weights."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from scipy.fft import rfft,irfft
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';CACHE=ROOT/'cache'
OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
SEED=71703
VARIANTS=['initial','full_update','norm_only','low_quarter','high_quarter','random_quarter','energy_quarter','gradient_quarter','random_norm_matched','random_spatial','gradient_shift_one']

def dump(p,x):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def hash_model(model):
    return {n:hashlib.sha256(memoryview(p.detach().contiguous().numpy()).cast('B')).hexdigest() for n,p in model.named_parameters()}

@torch.no_grad()
def teacher_cache(model,windows,name):
    path=CACHE/f'teacher_{name}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    hidden=[];nll=[]
    for i in range(0,len(windows),2):
        ids=windows[i:i+2];h=model.model(ids[:,:-1],use_cache=False).last_hidden_state
        logits=model.lm_head(h);loss=F.cross_entropy(logits.flatten(0,1),ids[:,1:].flatten(),reduction='none').reshape(len(ids),-1).mean(1)
        hidden.append(h);nll.append(loss)
    result={'hidden':torch.cat(hidden),'nll':torch.cat(nll)};torch.save(result,path);return result

def gradients(model,original,windows,reference,candidate,install,restore):
    blocks=candidate(original,'retained',trainable=True);install(model,original,blocks)
    losses=[];n_batches=(len(windows)+1)//2
    for i in range(0,len(windows),2):
        ids=windows[i:i+2]
        with torch.no_grad():lp=F.log_softmax(model.lm_head(reference['hidden'][i:i+2]),-1);prob=lp.exp()
        logits=model(ids[:,:-1],use_cache=False).logits;lq=F.log_softmax(logits,-1)
        loss=(prob*(lp-lq)).sum(-1).mean();losses.append(float(loss));(loss/n_batches).backward()
    result={n:p.grad.detach().clone().numpy() for n,p in blocks.named_parameters()}
    restore(model,original);del blocks;gc.collect()
    dump(OUT/'calibration.json',{'source':'first eight phase-6 training windows','targets':len(windows)*128,'mean_kl':float(np.mean(losses)),'optimizer_steps':0,'gradient_calls':n_batches})
    return result

def construct_masks(initial,final,grads):
    rows={};masks={v:{} for v in ['low_quarter','high_quarter','random_quarter','energy_quarter','gradient_quarter']};rng=np.random.default_rng(SEED)
    deltas={};initial_np={};final_np={};total_dof=0;counts={v:0 for v in masks};checks=[]
    for name,p in initial.named_parameters():
        base=p.detach().numpy().copy();target=dict(final.named_parameters())[name].detach().numpy().copy()
        initial_np[name]=base;final_np[name]=target;delta=target.astype(np.float64)-base.astype(np.float64);deltas[name]=delta
        total_dof+=delta.size
        if delta.ndim!=2:
            for v in counts:counts[v]+=delta.size
            continue
        nr,nc=delta.shape;assert nr%32==0 and nc%2==0;groups=nr//32
        fd=rfft(delta,axis=1,norm='ortho',workers=4);fg=rfft(grads[name].astype(np.float64),axis=1,norm='ortho',workers=4)
        # Equal contiguous partitions of positive bins except real endpoints.
        edges=np.linspace(1,fd.shape[1]-1,17,dtype=int)
        energy=np.empty((groups,16));descent=np.empty_like(energy)
        for b in range(16):
            lo,hi=edges[b:b+2]
            energy[:,b]=(2*abs(fd[:,lo:hi])**2).reshape(groups,32,-1).sum((1,2))
            descent[:,b]=(-2*(np.conj(fg[:,lo:hi])*fd[:,lo:hi]).real).reshape(groups,32,-1).sum((1,2))
        choices={
            'low_quarter':np.tile(np.arange(4),(groups,1)),
            'high_quarter':np.tile(np.arange(12,16),(groups,1)),
            'random_quarter':np.stack([np.sort(rng.choice(16,4,replace=False)) for _ in range(groups)]),
            'energy_quarter':np.sort(np.argsort(-energy,axis=1,kind='stable')[:,:4],axis=1),
            'gradient_quarter':np.sort(np.argsort(-descent,axis=1,kind='stable')[:,:4],axis=1)}
        for variant,choice in choices.items():
            masks[variant][name]={'edges':edges.tolist(),'bands':choice.tolist(),'row_group':32}
            counts[variant]+=int(2*nr+sum(64*(edges[b+1]-edges[b]) for group in choice for b in group))
        spectral_dot=float((2*(np.conj(fg)*fd).real).sum()-(np.conj(fg[:,0])*fd[:,0]).real.sum()-(np.conj(fg[:,-1])*fd[:,-1]).real.sum())
        direct_dot=float(np.sum(grads[name].astype(np.float64)*delta))
        checks.append({'name':name,'parseval_gradient_dot_abs_error':abs(spectral_dot-direct_dot),'roundtrip_max_error':float(np.max(abs(irfft(fd,n=nc,axis=1,norm='ortho')-delta)))})
        rows[name]={'shape':list(delta.shape),'energy':energy.tolist(),'descent':descent.tolist(),'edges':edges.tolist(),'update_energy':float(np.sum(delta**2)),'initial_energy':float(np.sum(base.astype(np.float64)**2)),'gradient_dot_update':direct_dot}
    dump(OUT/'local_regions.json',rows);dump(OUT/'masks.json',masks)
    dump(OUT/'transform_checks.json',{'checks':checks,'all_passed':max(r['roundtrip_max_error'] for r in checks)<1e-12 and max(r['parseval_gradient_dot_abs_error'] for r in checks)<1e-9})
    dump(OUT/'coefficient_budget.json',{'full_update_real_dof':total_dof,'counts_including_endpoints_and_norms':counts,'fraction':{v:n/total_dof for v,n in counts.items()},'seed':SEED,'semantic_warning':'Degrees of freedom of a post-hoc mask; these coefficients were not independently trained in this study.'})
    return initial_np,final_np,deltas,masks

def filtered(delta,definition):
    f=rfft(delta,axis=1,norm='ortho',workers=4);mask=np.zeros(f.shape,dtype=bool);mask[:,0]=True;mask[:,-1]=True
    edges=definition['edges']
    for g,bands in enumerate(definition['bands']):
        for b in bands:mask[g*32:(g+1)*32,edges[b]:edges[b+1]]=True
    f*=mask;return irfft(f,n=delta.shape[1],axis=1,norm='ortho',workers=4),int(2*mask.sum()-2*delta.shape[0])

def make_variant(original,variant,initial,final,deltas,masks,candidate):
    blocks=candidate(original,'retained');summaries={};rng=np.random.default_rng(SEED+1)
    with torch.no_grad():
        for name,p in blocks.named_parameters():
            delta=deltas[name];base=initial[name]
            if variant=='initial':p.copy_(torch.from_numpy(base));continue
            if variant=='full_update' or delta.ndim!=2:p.copy_(torch.from_numpy(final[name]));continue
            if variant=='norm_only':update=np.zeros_like(delta);dof=0
            elif variant in masks:update,dof=filtered(delta,masks[variant][name])
            elif variant=='gradient_shift_one':
                raw,dof=filtered(delta,masks['gradient_quarter'][name]);update=np.roll(raw,1,axis=1)
            elif variant=='random_norm_matched':
                raw,dof=filtered(delta,masks['random_quarter'][name]);target,_=filtered(delta,masks['gradient_quarter'][name])
                update=raw*np.sqrt(np.sum(target**2)/max(np.sum(raw**2),1e-30))
            elif variant=='random_spatial':
                _,dof=filtered(delta,masks['gradient_quarter'][name]);mask=np.zeros(delta.size,dtype=bool);mask[rng.choice(delta.size,dof,replace=False)]=True
                update=(delta.ravel()*mask).reshape(delta.shape)
            else:raise ValueError(variant)
            p.copy_(torch.from_numpy((base.astype(np.float64)+update).astype(np.float32)))
            summaries[name]={'retained_update_energy_fraction':float(np.sum(update**2)/np.sum(delta**2)),'real_dof':dof}
    return blocks,summaries

def install_depth(model,original,blocks,depth,install):
    if depth==28:return install(model,original,blocks)
    assert depth==26
    sparse=torch.nn.ModuleDict({s:torch.nn.ModuleList([blocks[s][0],blocks[s][2]]) for s in ['4','12']})
    return install(model,original,sparse)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase6',type=Path,required=True);args=ap.parse_args();src=args.phase6.resolve()
    os.environ['SMOLLM_MODEL_DIR']=str(src/'reference_model');sys.path.insert(0,str(src))
    from common import load_model
    from models import candidate,install,restore
    from evaluation import evaluate,cluster_ci
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    begin=time.perf_counter();model=load_model();original=list(model.model.layers);before=hash_model(model)
    with np.load(src/'data/token_windows.npz') as z:windows={k:torch.from_numpy(z[k].astype(np.int64)) for k in ['train','validation','audit']}
    windows['calibration']=windows['train'][:8]
    manifest=json.loads((src/'data/dataset_manifest.json').read_text())
    references={s:teacher_cache(model,windows[s],s) for s in ['calibration','validation','audit']}
    print('TEACHER_CACHES_COMPLETE',flush=True)
    grads=gradients(model,original,windows['calibration'],references['calibration'],candidate,install,restore)
    initial=candidate(original,'retained');final=candidate(original,'retained',checkpoint=src/'checkpoints/retained.safetensors')
    bases,targets,deltas,masks=construct_masks(initial,final,grads)
    del initial,final,grads;gc.collect()
    print('MASKS_LOCKED_BEFORE_EVALUATION',flush=True)
    rows=[];update_stats={}
    configurations=[(v,28) for v in VARIANTS]+[(v,26) for v in ['initial','full_update','gradient_quarter','random_quarter']]
    for variant,depth in configurations:
        blocks,stats=make_variant(original,variant,bases,targets,deltas,masks,candidate);update_stats[variant]=stats
        install_depth(model,original,blocks,depth,install);assert len(model.model.layers)==depth
        entry={'variant':variant,'depth':depth,'splits':{}}
        for split in ['validation','audit']:
            m=evaluate(model,windows[split],references[split]);assert m['all_finite']
            docids=[r['document_index'] for r in manifest[f'{split}_windows_manifest']]
            m['delta_nll_document_bootstrap_95']=cluster_ci(m['per_window_delta_nll'],docids)
            entry['splits'][split]=m
        rows.append(entry);dump(OUT/'interventions.json',rows)
        print(json.dumps({'variant':variant,'depth':depth,'validation_PPL_ratio':entry['splits']['validation']['perplexity_ratio_to_original'],'audit_PPL_ratio':entry['splits']['audit']['perplexity_ratio_to_original'],'audit_KL':entry['splits']['audit']['mean_kl']}),flush=True)
        restore(model,original);del blocks;gc.collect()
    dump(OUT/'update_stats.json',update_stats)
    expected=json.loads((src/'results/audit.json').read_text());verification={}
    for old,new in [('initial_retained','initial'),('trained_retained','full_update')]:
        target=next(r['metrics'] for r in expected if r['configuration']['id']==old)
        actual=next(r['splits']['audit'] for r in rows if r['variant']==new and r['depth']==28)
        errors={k:abs(target[k]-actual[k]) for k in ['nll','mean_kl','delta_nll']}
        assert max(errors.values())<2e-5,errors;verification[new]=errors
    assert hash_model(model)==before
    comparisons=[];docids=[r['document_index'] for r in manifest['audit_windows_manifest']]
    for depth in [28,26]:
        subset=[r for r in rows if r['depth']==depth];init=next(r for r in subset if r['variant']=='initial')['splits']['audit'];full=next(r for r in subset if r['variant']=='full_update')['splits']['audit']
        random=next(r for r in subset if r['variant']=='random_quarter')['splits']['audit']
        for row in subset:
            m=row['splits']['audit'];diff=np.array(m['per_window_nll'])-random['per_window_nll']
            comparisons.append({'variant':row['variant'],'depth':depth,'NLL_recovery_fraction':(init['nll']-m['nll'])/(init['nll']-full['nll']),'delta_nll_vs_random':float(diff.mean()),'paired_ci_vs_random':cluster_ci(diff,docids)})
    dump(OUT/'comparisons.json',comparisons)
    dump(OUT/'functional_checks.json',{'original_tensor_hashes_unchanged':True,'phase6_reproduction_abs_errors':verification,'total_seconds':time.perf_counter()-begin,'torch':torch.__version__,'threads':4,'calibration_targets':1024,'audit_targets':8192,'audit_documents':32,'new_optimizer_steps':0})
    print('INTERVENTIONS_COMPLETE',flush=True)

if __name__=='__main__':main()
