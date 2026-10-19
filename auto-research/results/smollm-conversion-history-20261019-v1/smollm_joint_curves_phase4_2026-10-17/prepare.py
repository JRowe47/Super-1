"""Verify an exact all-layer representation, and cache native teacher paths."""
import hashlib, json, shutil, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache
from common import ROOT,OUT,WINDOWS,dump,setup,load_model,hashes,data
from curve_model import HermiteBank,CurveBlock,restore,window_forward,path_loss


def select_data():
    if (ROOT/'data/token_windows.npz').exists():return
    src=ROOT.parent/'smollm_depth_curve_phase3/data'
    counts={'train':256,'validation':32,'test':64}
    with np.load(src/'token_windows.npz') as a:
        np.savez_compressed(ROOT/'data/token_windows.npz',**{k:a[k][:n] for k,n in counts.items()})
    prior=json.loads((src/'dataset_manifest.json').read_text())
    manifest={k:prior[k] for k in ['repository','config','revision','window_tokens','context_tokens']}
    manifest.update(counts=counts,selection='First N windows of each phase-3 split; no new random selection.',
        prior_manifest_sha256=hashlib.sha256((src/'dataset_manifest.json').read_bytes()).hexdigest(),
        limitation='Previously examined phase-3 corpus; test is held out from this optimization and checkpoint selection, but is not a fresh external benchmark.',
        splits={k:prior['splits'][k]['selected_windows'][:n] for k,n in counts.items()})
    dump(ROOT/'data/dataset_manifest.json',manifest)
    shutil.copy2(src/'dataset_manifest.json',ROOT/'reference/phase3_dataset_manifest.json')
    for name in ['MODEL_LICENSE.txt','MODEL_NOTICE.md']:
        shutil.copy2(src.parent/name,ROOT/name)


@torch.no_grad()
def cache_check(model,ids):
    full=model(ids,use_cache=False).logits
    cache=DynamicCache();out=[]
    for j in range(ids.shape[1]):
        out.append(model(ids[:,j:j+1],past_key_values=cache,use_cache=True).logits)
    diff=float((full-torch.cat(out,1)).abs().max())
    return {'max_cached_logit_abs_difference':diff,'cache_layers':len(cache),
        'kv_bytes':sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v in zip(cache.key_cache,cache.value_cache)),
        'tokens':ids.numel(),'tolerance':.001,'passed':diff<.001}


@torch.no_grad()
def exact(model,original,windows):
    before=hashes(model);ids=windows[:2,:-1];ref=model(ids,use_cache=False).logits
    native_cache=cache_check(model,ids[:1,:32])
    bank=HermiteBank(original,trainable=False)
    model.model.layers=nn.ModuleList([CurveBlock(model.config,bank,i,1,i) for i in range(30)])
    logits=model(ids,use_cache=False).logits;diff=float((logits-ref).abs().max())
    curved_cache=cache_check(model,ids[:1,:32]);restore(model,original)
    assert diff==0 and native_cache['passed'] and curved_cache['passed']
    after=hashes(model);assert before==after
    result={'all_30_original_knots_max_logit_abs_difference':diff,'prefill_tokens':ids.numel(),
        'native_cache':native_cache,'all_30_curve_cache':curved_cache,
        'original_tensor_hashes_unchanged':before==after,'unique_parameter_values':sum(p.numel() for p in model.parameters()),
        'families':{n:list(p.shape) for n,p in original[0].named_parameters()},
        'curve_knots':30,'exact_knot_rule':'Return original frozen tensor directly at each integer depth.',
        'original_parameter_sha256':before}
    dump(OUT/'exact_control.json',result);print('EXACT',diff,curved_cache,flush=True)


@torch.no_grad()
def cache_split(model,windows,label,include_paths):
    path=ROOT/'cache'/f'{label}.pt'
    if path.exists():return
    states={s:[] for s in WINDOWS};hidden=[];loss=[];handles=[];current={}
    if include_paths:
        for s in WINDOWS:
            handles.append(model.model.layers[s].register_forward_pre_hook(
                lambda m,a,s=s: current.__setitem__((s,0),a[0].detach())))
            for j in range(4):
                handles.append(model.model.layers[s+j].register_forward_hook(
                    lambda m,a,o,s=s,j=j: current.__setitem__((s,j+1),o[0].detach())))
    starttime=time.perf_counter()
    for start in range(0,len(windows),2):
        rows=windows[start:start+2]; h=model.model(rows[:,:-1],use_cache=False).last_hidden_state
        hidden.append(h.clone());logits=model.lm_head(h)
        loss.extend(F.cross_entropy(logits.flatten(0,1),rows[:,1:].flatten(),reduction='none').view(len(rows),-1).mean(1).tolist())
        if include_paths:
            for s in WINDOWS:states[s].append(torch.stack([current[(s,j)] for j in range(5)],1).clone())
    for handle in handles:handle.remove()
    result={'hidden':torch.cat(hidden),'nll':torch.tensor(loss,dtype=torch.float64)}
    for s in WINDOWS:
        if include_paths:result[f'window_{s}']=torch.cat(states[s])
    torch.save(result,path);print('CACHE',label,len(windows),round(time.perf_counter()-starttime,2),flush=True)


def pilot(model,original,cache):
    bank=HermiteBank(original[4:8]);teacher=cache['window_4'][:2]
    floor=float((cache['window_4'][:,-1]-cache['window_4'][:,0]).square().mean(-1).median())*.001
    optimizer=torch.optim.AdamW(bank.parameters(),lr=.001,weight_decay=0)
    values=[];start=time.perf_counter()
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        predictions=window_forward(model,bank,teacher[:,0],2)
        loss=path_loss(predictions,teacher,floor);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(bank.parameters(),1.)
        assert torch.isfinite(loss) and torch.isfinite(grad)
        optimizer.step();values.append(float(loss.detach()))
    result={'losses':values,'seconds':time.perf_counter()-start,'trainable_slope_values':sum(p.numel() for p in bank.parameters()),
        'all_gradients_finite':True,'excluded_from_main_training':True,'purpose':'Fixed-minibatch implementation and learning smoke test, not a quality result.'}
    dump(OUT/'pilot.json',result);print('PILOT',result,flush=True)


if __name__=='__main__':
    setup();select_data();windows=data();model=load_model();original=list(model.model.layers)
    exact(model,original,windows['validation'])
    for split in ['train','validation']:cache_split(model,windows[split],split,True)
    pilot(model,original,torch.load(ROOT/'cache/train.pt',weights_only=True))
