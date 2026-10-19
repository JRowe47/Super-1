"""Common first-block and full-model measurements for phase 3."""
import copy, hashlib, math, statistics, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
BATCH=2


def digest_tensor(t):
    return hashlib.sha256(memoryview(t.detach().cpu().contiguous().numpy()).cast('B')).hexdigest()


def call_block(model,block,ids):
    x=model.model.embed_tokens(ids);pos=torch.arange(ids.shape[1])[None]
    mask=torch.triu(torch.full((ids.shape[1],ids.shape[1]),torch.finfo(x.dtype).min),diagonal=1)[None,None]
    rope=model.model.rotary_emb(x,pos)
    return block(x,attention_mask=mask,position_ids=pos,position_embeddings=rope,use_cache=False)[0]


@torch.no_grad()
def reference_cache(model,windows,label):
    path=ROOT/'cache'/f'{label}_reference.pt'
    if path.exists():return torch.load(path,weights_only=True)
    first=[];handle=model.model.layers[0].register_forward_hook(lambda m,a,o:first.append(o[0].detach()))
    hs=[];bs=[];losses=[];ss=[]
    for start in range(0,len(windows),BATCH):
        rows=windows[start:start+BATCH];ids=rows[:,:-1];target=rows[:,1:]
        hidden=model.model(ids,use_cache=False).last_hidden_state;logits=model.lm_head(hidden)
        block=first.pop();hs.append(hidden.clone());bs.append(block.clone())
        losses.extend(F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='none').view(len(rows),-1).mean(1).tolist())
        ss.extend((block-model.model.embed_tokens(ids)).square().sum((1,2)).tolist())
    handle.remove();data={'hidden':torch.cat(hs),'block':torch.cat(bs),'nll':torch.tensor(losses,dtype=torch.float64),'residual_ss':torch.tensor(ss,dtype=torch.float64)}
    torch.save(data,path);return data


@torch.no_grad()
def evaluate(model,windows,reference):
    first=[];handle=model.model.layers[0].register_forward_hook(lambda m,a,o:first.append(o[0].detach()))
    losses=[];kls=[];agreements=[];ss=0.;max_logit=0.;max_block=0.;finite=True;starttime=time.perf_counter()
    for start in range(0,len(windows),BATCH):
        rows=windows[start:start+BATCH];ids=rows[:,:-1];target=rows[:,1:];n=len(rows)
        logits=model(ids,use_cache=False).logits
        gold=model.lm_head(reference['hidden'][start:start+n]);delta=first.pop()-reference['block'][start:start+n]
        ss+=float(delta.square().sum());max_block=max(max_block,float(delta.abs().max()))
        max_logit=max(max_logit,float((logits-gold).abs().max()))
        losses.extend(F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='none').view(n,-1).mean(1).tolist())
        lp=F.log_softmax(gold,-1);lq=F.log_softmax(logits,-1)
        kls.extend((lp.exp()*(lp-lq)).sum(-1).mean(1).tolist())
        agreements.extend((logits.argmax(-1)==gold.argmax(-1)).float().mean(1).tolist())
        finite=finite and bool(torch.isfinite(logits).all())
    handle.remove();nll=np.asarray(losses);base=reference['nll'].numpy();diff=nll-base
    return {'windows':len(windows),'predicted_tokens':len(windows)*128,'nll':float(nll.mean()),'reference_nll':float(base.mean()),
      'delta_nll':float(diff.mean()),'perplexity_ratio_to_original':float(np.exp(diff.mean())),
      'mean_kl':float(np.mean(kls)),'top1_agreement':float(np.mean(agreements)),
      'block_residual_relative_rms_error':math.sqrt(ss/float(reference['residual_ss'].sum())),
      'max_block_abs':max_block,'max_logit_abs':max_logit,'all_finite':finite,
      'per_window_nll':nll.tolist(),'per_window_delta_nll':diff.tolist(),'evaluation_seconds':time.perf_counter()-starttime}


def cluster_ci(deltas,docids):
    ids=np.asarray(docids);d=np.asarray(deltas);unique=np.unique(ids)
    sums=np.array([d[ids==i].sum() for i in unique]);counts=np.array([(ids==i).sum() for i in unique])
    idx=np.random.default_rng(30416).integers(0,len(unique),(2000,len(unique)))
    means=sums[idx].sum(1)/counts[idx].sum(1)
    return {'lower':float(np.quantile(means,.025)),'upper':float(np.quantile(means,.975)),
            'clusters':len(unique),'bootstrap_replicates':2000,'unit':'document'}


@torch.no_grad()
def paired_timing(model,original,candidate,ids,pairs=12):
    result={}
    for scope in ['block','full_model']:
        def run(block):
            if scope=='block':return call_block(model,block,ids)
            model.model.layers[0]=block
            return model(ids,use_cache=False).logits
        for _ in range(3):run(original);run(candidate)
        measurements=[]
        for i in range(pairs):
            order=[('original',original),('candidate',candidate)]
            if i%2:order.reverse()
            row={}
            for name,block in order:
                begin=time.perf_counter();run(block);row[name]=(time.perf_counter()-begin)*1000
            measurements.append(row)
        result[scope]={'original_median_ms':statistics.median(r['original'] for r in measurements),
           'candidate_median_ms':statistics.median(r['candidate'] for r in measurements),
           'median_paired_ratio':statistics.median(r['candidate']/r['original'] for r in measurements),'pairs_ms':measurements}
    model.model.layers[0]=original
    return result
