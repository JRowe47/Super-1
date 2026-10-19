"""Full-model quality metrics and paired document bootstrap."""
import time
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate(model, windows, reference):
    losses=[];kls=[];agreements=[];max_logit=0.;finite=True;starttime=time.perf_counter()
    for start in range(0,len(windows),2):
        rows=windows[start:start+2];n=len(rows)
        logits=model.lm_head(model.model(rows[:,:-1],use_cache=False).last_hidden_state[:,-128:])
        gold=model.lm_head(reference['hidden'][start:start+n])
        max_logit=max(max_logit,float((logits-gold).abs().max()))
        losses.extend(F.cross_entropy(logits.flatten(0,1),rows[:,-128:].flatten(),reduction='none').view(n,-1).mean(1).tolist())
        lp=F.log_softmax(gold,-1);lq=F.log_softmax(logits,-1)
        kls.extend((lp.exp()*(lp-lq)).sum(-1).mean(1).tolist())
        agreements.extend((logits.argmax(-1)==gold.argmax(-1)).float().mean(1).tolist())
        finite=finite and bool(torch.isfinite(logits).all())
    nll=np.asarray(losses);base=reference['nll'][:len(windows)].numpy();diff=nll-base
    return {'windows':len(windows),'predicted_tokens':len(windows)*128,'context_tokens':windows.shape[1]-1,
      'nll':float(nll.mean()),'reference_nll':float(base.mean()),'delta_nll':float(diff.mean()),
      'perplexity_ratio_to_original':float(np.exp(diff.mean())),
      'mean_kl':float(np.mean(kls)),'top1_agreement':float(np.mean(agreements)),
      'max_logit_abs':max_logit,'all_finite':finite,'per_window_nll':nll.tolist(),
      'per_window_delta_nll':diff.tolist(),'per_window_kl':kls,
      'evaluation_seconds':time.perf_counter()-starttime}


def cluster_ci(deltas,docids):
    ids=np.asarray(docids);d=np.asarray(deltas);unique=np.unique(ids)
    sums=np.array([d[ids==i].sum() for i in unique]);counts=np.array([(ids==i).sum() for i in unique])
    idx=np.random.default_rng(50518).integers(0,len(unique),(2000,len(unique)))
    means=sums[idx].sum(1)/counts[idx].sum(1)
    return {'lower':float(np.quantile(means,.025)),'upper':float(np.quantile(means,.975)),
            'clusters':len(unique),'bootstrap_replicates':2000,'unit':'document'}
