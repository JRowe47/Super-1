"""Exploratory control added after primary results: equally guided spatial mask."""
import argparse,gc,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from interventions import ROOT,OUT,CACHE,SEED,dump,gradients,hash_model

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase6',type=Path,required=True);args=ap.parse_args();src=args.phase6.resolve()
    dump(ROOT/'protocol_addendum.json',{'status':'Exploratory addendum after all primary outcomes were available','reason':'Separate the value of loss-based targeting from Fourier representation; a random spatial control does not do this.','rule':'Use the same initial-model gradient on the same first eight training windows and the same dense learned update. Within each 32-row matrix group, select individual entries by largest -gradient times update. Retain exactly as many real weight coordinates as the corresponding primary spectral mask degrees of freedom. Preserve the same full norm updates.','evaluation':'One additional 28-block candidate on the existing validation and audit; no hyperparameter choices based on its outcomes.','limits':'Uses the known full dense solution, is retrospective, and remains conditional on one source seed and reused audit.'})
    os.environ['SMOLLM_MODEL_DIR']=str(src/'reference_model');sys.path.insert(0,str(src))
    from common import load_model
    from models import candidate,install,restore
    from evaluation import evaluate,cluster_ci
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();model=load_model();original=list(model.model.layers);before=hash_model(model)
    with np.load(src/'data/token_windows.npz') as z:windows={k:torch.from_numpy(z[k].astype(np.int64)) for k in ['train','validation','audit']}
    ref={k:torch.load(CACHE/f'teacher_{k}.pt',weights_only=True) for k in ['calibration','validation','audit']}
    grads=gradients(model,original,windows['train'][:8],ref['calibration'],candidate,install,restore)
    blocks=candidate(original,'retained');final=candidate(original,'retained',checkpoint=src/'checkpoints/retained.safetensors')
    targets=dict(final.named_parameters());masks=json.loads((OUT/'masks.json').read_text())['gradient_quarter'];kept=0;total=0;stats={}
    with torch.no_grad():
        for name,p in blocks.named_parameters():
            total+=p.numel();target=targets[name]
            if p.ndim!=2:p.copy_(target);kept+=p.numel();continue
            base=p.numpy().astype(np.float64);delta=target.numpy().astype(np.float64)-base;score=-grads[name].astype(np.float64)*delta
            definition=masks[name];edges=definition['edges'];update=np.zeros_like(delta);budget=0
            for group,bands in enumerate(definition['bands']):
                count=64+sum(64*(edges[b+1]-edges[b]) for b in bands)
                s=score[group*32:(group+1)*32].ravel();chosen=np.argpartition(s,len(s)-count)[-count:]
                part=update[group*32:(group+1)*32].reshape(-1);part[chosen]=delta[group*32:(group+1)*32].ravel()[chosen]
                budget+=count
            p.copy_(torch.from_numpy((base+update).astype(np.float32)));kept+=budget
            stats[name]={'real_dof':budget,'energy_fraction':float(np.sum(update**2)/np.sum(delta**2))}
    expected=json.loads((OUT/'coefficient_budget.json').read_text())['counts_including_endpoints_and_norms']['gradient_quarter'];assert kept==expected
    install(model,original,blocks);manifest=json.loads((src/'data/dataset_manifest.json').read_text());entry={'variant':'gradient_spatial','depth':28,'exploratory':True,'splits':{},'real_dof':kept,'fraction':kept/total}
    for split in ['validation','audit']:
        m=evaluate(model,windows[split],ref[split]);assert m['all_finite'];ids=[r['document_index'] for r in manifest[f'{split}_windows_manifest']]
        m['delta_nll_document_bootstrap_95']=cluster_ci(m['per_window_delta_nll'],ids);entry['splits'][split]=m
    primary=json.loads((OUT/'interventions.json').read_text());spectral=next(r['splits']['audit'] for r in primary if r['variant']=='gradient_quarter' and r['depth']==28)
    difference=np.array(entry['splits']['audit']['per_window_nll'])-spectral['per_window_nll'];ids=[r['document_index'] for r in manifest['audit_windows_manifest']]
    entry['delta_nll_vs_gradient_spectral']=float(difference.mean());entry['paired_ci_vs_gradient_spectral']=cluster_ci(difference,ids)
    init=next(r['splits']['audit'] for r in primary if r['variant']=='initial' and r['depth']==28);full=next(r['splits']['audit'] for r in primary if r['variant']=='full_update' and r['depth']==28)
    entry['NLL_recovery_fraction']=(init['nll']-entry['splits']['audit']['nll'])/(init['nll']-full['nll'])
    restore(model,original);assert before==hash_model(model);entry['original_hashes_unchanged']=True;entry['seconds']=time.perf_counter()-started
    dump(OUT/'targeting_control.json',entry);dump(OUT/'targeting_control_update_stats.json',stats)
    print(json.dumps({'audit_PPL_ratio':entry['splits']['audit']['perplexity_ratio_to_original'],'recovery_fraction':entry['NLL_recovery_fraction'],'delta_nll_vs_gradient_spectral':entry['delta_nll_vs_gradient_spectral'],'CI':entry['paired_ci_vs_gradient_spectral'],'seconds':entry['seconds']}),flush=True)

if __name__=='__main__':main()
