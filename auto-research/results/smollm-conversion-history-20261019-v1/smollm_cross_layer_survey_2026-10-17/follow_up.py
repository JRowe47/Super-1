"""Exploratory robustness and adjacent-pair diagnostics after the initial survey."""
from pathlib import Path
import json,time,resource
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from run_survey import ROOT,OUT,dump,hashes,model_directory,SkipBlock,linear_cka


@torch.no_grad()
def losses(model,windows):
    result=[]
    for start in range(0,len(windows),2):
        w=windows[start:start+2];logits=model(w[:,:-1],use_cache=False).logits
        result.extend(F.cross_entropy(logits.flatten(0,1),w[:,1:].flatten(),reduction='none').view(len(w),-1).mean(1).tolist())
    return np.asarray(result)


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();data=np.load(OUT/'representations.npz')
    matrices={};energy={}
    for kind,key in [('state','projected_sample_states'),('update','projected_sample_updates')]:
        x=torch.from_numpy(data[key]).double();squares=x.square().sum(-1)
        y=F.normalize(x,dim=-1)
        matrices[kind+'_unit_token_linear_cka']=linear_cka(y,torch.arange(x.shape[1])).numpy()
        energy[kind+'_top_3_of_256_tokens_energy_fraction']=(squares.topk(3,dim=1).values.sum(1)/squares.sum(1)).tolist()
    np.savez_compressed(OUT/'normalized_similarity.npz',**matrices)
    summary={k+'_median_adjacent':float(np.median(np.diag(v,1))) for k,v in matrices.items()}
    summary.update(energy)
    summary['note']='Post-hoc robustness check: normalize each sampled token vector to unit length before centering for linear CKA. Same 256 token indices as initial survey.'
    dump('normalized_similarity.json',summary)
    # Chosen after the initial survey to cover early/middle/late depth and the
    # large activation transition near block 11, not a held-out selection.
    pairs=[(2,3),(6,7),(10,11),(14,15),(20,21),(28,29)]
    dump('pair_protocol.json',{'pairs':pairs,'operations':['swap_order','remove_both'],
      'selection':'Exploratory follow-up chosen after initial survey; cover depth and the block-11 transition.',
      'data':'Same 1024-token validation probe; no independent generalization claim.',
      'interaction':'delta_nll_remove_both - delta_nll_remove_a - delta_nll_remove_b',
      'scope':'Each pair changed separately in the full model, then restored; cache disabled.'})
    model=AutoModelForCausalLM.from_pretrained(model_directory(),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    model.requires_grad_(False);before=hashes(model);blocks=list(model.model.layers)
    w=torch.from_numpy(np.load(ROOT/'data/probe.npz')['windows']).long();baseline=losses(model,w)
    single={r['removed_block']:r for r in json.loads((OUT/'single_block_ablation.json').read_text())}
    result=[]
    for a,b in pairs:
        for operation in ['swap_order','remove_both']:
            if operation=='swap_order':model.model.layers[a],model.model.layers[b]=blocks[b],blocks[a]
            else:model.model.layers[a],model.model.layers[b]=SkipBlock(),SkipBlock()
            try:changed=losses(model,w)
            finally:model.model.layers[a],model.model.layers[b]=blocks[a],blocks[b]
            delta=float((changed-baseline).mean())
            row={'pair':[a,b],'operation':operation,'delta_nll':delta,'perplexity_ratio':float(np.exp(delta)),
                 'per_window_delta_nll':(changed-baseline).tolist()}
            if operation=='remove_both':row['interaction_delta_nll']=delta-single[a]['delta_nll']-single[b]['delta_nll']
            result.append(row);dump('pair_results.json',result)
            print('PAIR',a,b,operation,'dNLL',round(delta,6),'PPLx',round(row['perplexity_ratio'],4),flush=True)
    assert hashes(model)==before and all(model.model.layers[i] is blocks[i] for i in range(30))
    difference=float(np.abs(losses(model,w)-baseline).max());assert difference==0
    dump('follow_up_summary.json',{'status':'PASS','all_parameters_unchanged':True,'restored_max_nll_difference':difference,
         'elapsed_seconds':time.perf_counter()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         'normalized_state_cka_median_adjacent':summary['state_unit_token_linear_cka_median_adjacent'],
         'normalized_update_cka_median_adjacent':summary['update_unit_token_linear_cka_median_adjacent']})
    print('FOLLOW-UP COMPLETE',flush=True)


if __name__=='__main__':main()
