"""Layer/family spectra and valid gated-MLP neuron permutation controls."""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.fft import rfft
import torch
from torch.nn import functional as F
from global_probe import Source, tree, dump
from interventions import hash_model, filtered

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
SEED=71705

def profile(x):
    x=np.asarray(x,dtype=np.float32).ravel()
    f=rfft(x,norm='ortho',workers=4)
    full=tree(f,len(x))
    levels={str(k):[n for n in full['nodes'] if n['level']==k] for k in range(1,7)}
    return {'n':len(x),'energy':full['total_energy'],
            'levels':{k:{'fraction':[n['fraction'] for n in row],
                         'real_dof':[n['real_dof'] for n in row]} for k,row in levels.items()}}

def tensors(source,p,arm,trained=False,delta=False):
    values={n:source.tensor(p,n,arm,trained) for n in source.names}
    if delta:
        values={n:source.tensor(p,n,arm,True)-v for n,v in values.items()}
    return values

def profiles(values):
    return {'block':profile(np.concatenate([v.ravel() for v in values.values()])),
            'families':{n:profile(v) for n,v in values.items()}}

def permute(values,p):
    out=dict(values)
    for name in ['mlp.gate_proj.weight','mlp.up_proj.weight']:out[name]=values[name][p,:]
    out['mlp.down_proj.weight']=values['mlp.down_proj.weight'][:,p]
    return out

def compare(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    fa=rfft(a,norm='ortho',workers=4);fb=rfft(b,norm='ortho',workers=4)
    weights=np.full(len(fa),2.);weights[0]=1
    if len(a)%2==0:weights[-1]=1
    pa=weights*abs(fa)**2;pb=weights*abs(fb)**2
    pa/=pa.sum();pb/=pb.sum()
    edges=np.linspace(0,len(fa),65,dtype=int)
    ca=np.array([pa[l:h].sum() for l,h in zip(edges[:-1],edges[1:])])
    cb=np.array([pb[l:h].sum() for l,h in zip(edges[:-1],edges[1:])])
    return {'complex_spectrum_relative_l2_difference':float(np.sqrt(np.sum(weights*abs(fb-fa)**2)/np.sum(weights*abs(fa)**2))),
            'fine_power_total_variation':float(.5*np.sum(abs(pa-pb))),
            'coarse_64_power_total_variation':float(.5*np.sum(abs(ca-cb))),
            'relative_energy_difference':float(abs(np.sum(a*a)-np.sum(b*b))/np.sum(a*a)),
            'native_64_fractions':ca.tolist(),'permuted_64_fractions':cb.tolist()}

def spectra(src):
    source=Source(src);records=[];permutations={};controls=[];rng=np.random.default_rng(SEED)
    masks=json.loads((OUT/'masks.json').read_text())['gradient_quarter']
    for block,p in enumerate(source.map):
        final=tensors(source,p,'retained',True)
        row={'block28':block,'source':p,'original_layer':p if isinstance(p,int) else p[0]+[0,2,3][p[1]],
             'variants':{'retained_final':profiles(final)}}
        if isinstance(p,tuple):
            for label,arm,trained,delta in [('retained_initial','retained',False,False),('retained_delta','retained',False,True),('interpolated_initial','interpolated',False,False),('interpolated_final','interpolated',True,False),('interpolated_delta','interpolated',False,True)]:
                row['variants'][label]=profiles(tensors(source,p,arm,trained,delta))
            perm=rng.permutation(final['mlp.gate_proj.weight'].shape[0]);permutations[f'{p[0]}.{p[1]}']=perm.tolist()
            delta=tensors(source,p,'retained',False,True)
            control={'block28':block,'source':p,'original_layer':row['original_layer'],'states':{}}
            for state,values in [('retained_final',final),('retained_delta',delta)]:
                changed=permute(values,perm)
                assert all(np.array_equal(v,permute(changed,np.argsort(perm))[n]) for n,v in values.items())
                control['states'][state]={'block':compare(np.concatenate([v.ravel() for v in values.values()]),np.concatenate([v.ravel() for v in changed.values()])),
                    'families':{n:compare(values[n],changed[n]) for n in ['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']}}
            name='mlp.down_proj.weight';definition=masks[f'{p[0]}.{p[1]}.{name}']
            native,_=filtered(delta[name].astype(np.float64),definition)
            changed,_=filtered(delta[name][:,perm].astype(np.float64),definition)
            back=changed[:,np.argsort(perm)]
            control['fixed_down_mask_relative_difference_after_inverse_mapping']=float(np.linalg.norm(back-native)/np.linalg.norm(native))
            controls.append(control)
        records.append(row)
        print(json.dumps({'block28':block,'original_layer':row['original_layer'],'states':len(row['variants'])}),flush=True)
    dump(OUT/'layer_family_spectra.json',records)
    dump(OUT/'permutation_indices.json',permutations)
    dump(OUT/'permutation_spectral_controls.json',controls)
    return permutations

@torch.no_grad()
def functional(src,permutations):
    os.environ['SMOLLM_MODEL_DIR']=str(src/'reference_model');sys.path.insert(0,str(src))
    from common import load_model
    from models import candidate,install,restore
    model=load_model();original=list(model.model.layers);original_hash=hash_model(model)
    blocks=candidate(original,'retained',checkpoint=src/'checkpoints/retained.safetensors')
    block_hash=hash_model(blocks);install(model,original,blocks)
    with np.load(src/'data/token_windows.npz') as z:ids=torch.from_numpy(z['train'][:8].astype(np.int64))
    before=torch.cat([model.model(ids[i:i+2,:-1],use_cache=False).last_hidden_state for i in range(0,len(ids),2)])
    def apply(inverse=False):
        for key,indices in permutations.items():
            s,j=key.split('.');block=blocks[s][int(j)];p=torch.tensor(indices)
            if inverse:p=torch.argsort(p)
            for n in ['gate_proj','up_proj']:
                w=getattr(block.mlp,n).weight;w.copy_(w[p,:])
            w=block.mlp.down_proj.weight;w.copy_(w[:,p])
    apply();maximum=0.;err_sq=0.;ref_sq=0.;nll0=[];nll1=[];agree=0;total=0
    for i in range(0,len(ids),2):
        h=model.model(ids[i:i+2,:-1],use_cache=False).last_hidden_state
        a=model.lm_head(before[i:i+2]);b=model.lm_head(h);d=(b-a).double()
        maximum=max(maximum,float(d.abs().max()));err_sq+=float((d*d).sum());ref_sq+=float((a.double()**2).sum())
        agree+=int((a.argmax(-1)==b.argmax(-1)).sum());total+=a.shape[0]*a.shape[1]
        target=ids[i:i+2,1:].flatten()
        nll0.append(float(F.cross_entropy(a.flatten(0,1),target)));nll1.append(float(F.cross_entropy(b.flatten(0,1),target)))
    apply(True);assert hash_model(blocks)==block_hash
    restore(model,original);assert hash_model(model)==original_hash
    relative=(err_sq/ref_sq)**.5;difference=float(np.mean(nll1)-np.mean(nll0))
    assert maximum<2e-3 and relative<1e-5 and abs(difference)<1e-5
    result={'seed':SEED,'permuted_layers':len(permutations),'targets':total,'source':'first eight training windows, no evaluation-set tuning',
            'maximum_absolute_logit_difference':maximum,'relative_rms_logit_difference':relative,'argmax_agreement':agree/total,
            'nll_difference':difference,'candidate_inverse_permutation_hash_exact':True,'original_hashes_unchanged':True,
            'algebraic_identity':'gate/up rows permuted together; down columns receive same permutation; all external block coordinates unchanged'}
    dump(OUT/'permutation_functional_checks.json',result)
    print(json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase6',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();permutations=spectra(args.phase6.resolve());gc.collect();functional(args.phase6.resolve(),permutations)
    dump(OUT/'layer_permutation_environment.json',{'total_seconds':time.perf_counter()-started,'seed':SEED,'primary_source':str(args.phase6.resolve()),'optimizer_steps':0})

if __name__=='__main__':main()
