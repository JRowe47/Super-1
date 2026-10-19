"""Read-only 30-block survey; small exploratory validation probe, not a benchmark."""
from pathlib import Path
import hashlib, json, math, resource, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
REVISION='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def model_directory():
    local=ROOT/'reference_model'
    return local if (local/'model.safetensors').exists() else ROOT.parent/'smollm_first_layer_curve/reference_model'


def hashes(model):
    return {n:hashlib.sha256(memoryview(p.detach().contiguous().numpy()).cast('B')).hexdigest() for n,p in model.named_parameters()}


def cosine_matrix(x):
    flat=x.flatten(1).double();norm=flat.square().sum(1).sqrt()
    return ((flat@flat.T)/norm[:,None]/norm[None,:]).clamp(-1,1)


def linear_cka(x,indices):
    # Feature centering implements HKH. Biased linear CKA, same token samples.
    grams=[]
    for layer in x:
        y=layer[indices].double();y=y-y.mean(0,keepdim=True)
        grams.append((y@y.T).flatten())
    return cosine_matrix(torch.stack(grams))


@torch.no_grad()
def weights(model):
    keys=list(dict(model.model.layers[0].named_parameters()))
    rows=[];total=torch.zeros(30,30,dtype=torch.float64)
    for key in keys:
        x=torch.stack([dict(b.named_parameters())[key].flatten() for b in model.model.layers])
        gram=(x@x.T).double();total+=gram
        norm=gram.diag().sqrt();cos=gram/norm[:,None]/norm[None,:]
        h=torch.eye(30,dtype=torch.float64)-torch.ones(30,30,dtype=torch.float64)/30
        eig=torch.linalg.eigvalsh(h@gram@h).clamp_min(0).flip(0)
        cumulative=eig.cumsum(0)/eig.sum()
        rows.append({'tensor':key,'shape':list(dict(model.model.layers[0].named_parameters())[key].shape),
          'adjacent_cosine_mean':float(cos.diagonal(1).mean()),'adjacent_cosine_min':float(cos.diagonal(1).min()),
          'centered_depth_rank_for_90_percent_energy':int(torch.searchsorted(cumulative,torch.tensor(.9))+1),
          'centered_depth_cumulative_energy':cumulative.tolist()})
        print('WEIGHTS',key,rows[-1]['adjacent_cosine_mean'],flush=True)
    norm=total.diag().sqrt();cos=total/norm[:,None]/norm[None,:]
    interp=[]
    for i in range(1,29):
        a=torch.zeros(30,dtype=torch.float64);a[i]=1;a[i-1]=a[i+1]=-.5
        interp.append(float((a@total@a/total[i,i]).clamp_min(0).sqrt()))
    return {'tensors':rows,'all_weight_cosine':cos.tolist(),'adjacent_weight_cosine':cos.diagonal(1).tolist(),
            'neighbor_average_relative_weight_error':interp,
            'note':'Raw coordinates; no neuron/head alignment. This does not measure functional equivalence.'}


class SkipBlock(nn.Module):
    def forward(self,hidden_states,*args,**kwargs):
        assert not kwargs.get('use_cache',False)
        assert not kwargs.get('output_attentions',False)
        return (hidden_states,)


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(40417)
    torch.use_deterministic_algorithms(True);started=time.perf_counter()
    ids=torch.from_numpy(np.load(ROOT/'data/probe.npz')['windows']).long()
    protocol={'scope':'All 30 original blocks. No training or curve fitting. One-block removal at a time.',
      'model':'HuggingFaceTB/SmolLM2-135M','revision':REVISION,'windows':len(ids),'context':128,
      'predicted_tokens':len(ids)*128,'data':'First 8 phase-3 validation windows, reused for exploratory diagnosis; no fresh held-out claim.',
      'batch':2,'dtype':'float32','threads':4,'attention':'eager','cache':False,
      'cka':'Biased centered linear CKA on 256 evenly indexed probe tokens; descriptive, dependent tokens.',
      'ablation':'Every original block 0..29 separately replaced by identity. No selection of a compressed model.'}
    dump('protocol.json',protocol)
    model=AutoModelForCausalLM.from_pretrained(model_directory(),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    model.requires_grad_(False);before=hashes(model)
    wr=weights(model);dump('weights.json',wr)
    blocks=list(model.model.layers);inputs=[[] for _ in blocks];outputs=[[] for _ in blocks]
    attention=[[] for _ in blocks];mlp=[[] for _ in blocks];handles=[]
    def capture(i):
        def hook(mod,args,out):
            inputs[i].append(args[0].detach().flatten(0,1).clone())
            outputs[i].append(out[0].detach().flatten(0,1).clone())
        return hook
    def capture_component(storage,i,tuple_output=False):
        def hook(mod,args,out):
            if tuple_output:out=out[0]
            storage[i].append(out.detach().flatten(0,1).clone())
        return hook
    for i,b in enumerate(blocks):
        handles.extend([b.register_forward_hook(capture(i)),b.self_attn.register_forward_hook(capture_component(attention,i,True)),
                        b.mlp.register_forward_hook(capture_component(mlp,i))])
    base_nll=[];base_argmax=[];baseline_logits=None;begin=time.perf_counter()
    for start in range(0,len(ids),2):
        x=ids[start:start+2];logits=model(x[:,:-1],use_cache=False).logits
        if start==0:baseline_logits=logits.clone()
        base_nll.extend(F.cross_entropy(logits.flatten(0,1),x[:,1:].flatten(),reduction='none').view(len(x),-1).mean(1).tolist())
        base_argmax.append(logits.argmax(-1))
    probe_seconds=time.perf_counter()-begin
    for h in handles:h.remove()
    x=torch.stack([torch.cat(z) for z in inputs]);y=torch.stack([torch.cat(z) for z in outputs]);delta=y-x
    att=torch.stack([torch.cat(z) for z in attention]);ff=torch.stack([torch.cat(z) for z in mlp])
    del inputs,outputs,attention,mlp
    states=torch.cat([x[0:1],y],dim=0);indices=torch.linspace(0,len(ids)*128-1,256).long()
    state_cka=linear_cka(states,indices);update_cka=linear_cka(delta,indices)
    update_cos=cosine_matrix(delta)
    adjacent_token_cos=F.cosine_similarity(delta[:-1],delta[1:],dim=-1).mean(-1)
    residual_cos=F.cosine_similarity(x,y,dim=-1).mean(-1)
    rows=[]
    for i in range(30):
        rows.append({'block':i,'mean_input_output_cosine':float(residual_cos[i]),
          'residual_update_rms_over_input_rms':float((delta[i].square().sum()/x[i].square().sum()).sqrt()),
          'attention_update_rms_over_input_rms':float((att[i].square().sum()/x[i].square().sum()).sqrt()),
          'mlp_update_rms_over_input_rms':float((ff[i].square().sum()/x[i].square().sum()).sqrt()),
          'attention_mlp_update_cosine':float(F.cosine_similarity(att[i],ff[i],dim=-1).mean()),
          'adjacent_update_mean_token_cosine':float(adjacent_token_cos[i]) if i<29 else None})
    ar={'blocks':rows,'state_linear_cka':state_cka.tolist(),'update_linear_cka':update_cka.tolist(),
        'update_global_cosine':update_cos.tolist(),
        'note':'Updates are measured on each block\'s natural inputs along the same token trajectories, not on matched inputs.'}
    dump('activations.json',ar)
    np.savez_compressed(OUT/'representations.npz',state_linear_cka=state_cka.numpy(),update_linear_cka=update_cka.numpy(),
      update_global_cosine=update_cos.numpy(),cka_token_indices=indices.numpy(),
      projected_sample_states=states[:,indices,:].numpy(),projected_sample_updates=delta[:,indices,:].numpy())
    del x,y,delta,att,ff,states
    base=np.asarray(base_nll);gold_argmax=torch.cat(base_argmax);ablations=[]
    for i,original in enumerate(blocks):
        nll=[];agreement=[];model.model.layers[i]=SkipBlock()
        try:
            for start in range(0,len(ids),2):
                w=ids[start:start+2];logits=model(w[:,:-1],use_cache=False).logits
                nll.extend(F.cross_entropy(logits.flatten(0,1),w[:,1:].flatten(),reduction='none').view(len(w),-1).mean(1).tolist())
                agreement.append(float((logits.argmax(-1)==gold_argmax[start:start+len(w)]).float().mean()))
        finally:model.model.layers[i]=original
        diff=np.asarray(nll)-base
        row={'removed_block':i,'delta_nll':float(diff.mean()),'perplexity_ratio':float(np.exp(diff.mean())),
             'argmax_agreement':float(np.mean(agreement)),'per_window_delta_nll':diff.tolist()}
        ablations.append(row);dump('single_block_ablation.json',ablations)
        print('SKIP',i,'dNLL',round(row['delta_nll'],6),'PPLx',round(row['perplexity_ratio'],4),flush=True)
    assert all(model.model.layers[i] is blocks[i] for i in range(30))
    assert hashes(model)==before
    restored=model(ids[:2,:-1],use_cache=False).logits
    restoration_error=float((restored-baseline_logits).abs().max());assert restoration_error==0
    summary={'status':'PASS','model_parameters':sum(p.numel() for p in model.parameters()),'blocks':30,
      'reference_nll':float(base.mean()),'reference_perplexity':float(np.exp(base.mean())),
      'probe_predicted_tokens':len(ids)*128,'baseline_probe_seconds_with_hooks':probe_seconds,
      'elapsed_seconds':time.perf_counter()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      'parameter_tensors_checked':len(before),'all_parameters_unchanged':True,'restored_max_logit_difference':restoration_error,
      'median_adjacent_state_cka':float(torch.quantile(state_cka.diagonal(1),.5)),
      'median_adjacent_update_cka':float(update_cka.diagonal(1).median()),
      'median_adjacent_update_mean_token_cosine':float(adjacent_token_cos.median()),
      'median_adjacent_raw_weight_cosine':float(np.median(wr['adjacent_weight_cosine'])),
      'median_neighbor_average_relative_weight_error':float(np.median(wr['neighbor_average_relative_weight_error'])),
      'lowest_single_skip_delta_nll':min(ablations,key=lambda r:r['delta_nll']),
      'highest_single_skip_delta_nll':max(ablations,key=lambda r:r['delta_nll'])}
    dump('summary.json',summary);print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
