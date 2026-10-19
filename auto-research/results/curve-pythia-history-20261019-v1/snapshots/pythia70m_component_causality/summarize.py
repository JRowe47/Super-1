"""Paired confirmation statistics and full-vocabulary component interactions."""
import json
from pathlib import Path
import numpy as np
from causal_probe import ROOT,OUT,RAW,read,dump,project

def boot_indices(seqrows):
    rng=np.random.default_rng(2026101704);groups=[]
    for n in sorted({r['n'] for r in seqrows}):
        ix=np.array([i for i,r in enumerate(seqrows) if r['n']==n])
        groups.append(rng.choice(ix,(4000,len(ix)),replace=True))
    return np.concatenate(groups,axis=1)

def paired(values,ix,weights=None):
    v=np.asarray(values,dtype=np.float64)
    if weights is None:point=v.mean();samples=v[ix].mean(1)
    else:
        w=np.asarray(weights,dtype=np.float64);point=np.dot(v,w)/sum(w);samples=np.sum(v[ix]*w[ix],axis=1)/np.sum(w[ix],axis=1)
    return dict(estimate=float(point),ci95=np.quantile(samples,[.025,.975]).tolist())

def summarize_cases(rows):
    ans=[]
    for phase in sorted({r['phase'] for r in rows}):
        base=next(r for r in rows if r['phase']==phase and r['id']=='baseline')
        ix=boot_indices(base['repeat']['sequences'])
        pidx=np.random.default_rng(2026101705).integers(0,len(base['prose']['passages']),(4000,len(base['prose']['passages'])))
        for r in [x for x in rows if x['phase']==phase and x['id']!='baseline']:
            result=dict(id=r['id'],phase=phase,update_norm=r.get('update_norm'),by_length={})
            for kind in ['repeat','scrambled']:
                a=base[kind]['sequences'];b=r[kind]['sequences'];assert [v['id'] for v in a]==[v['id'] for v in b]
                result[kind]={'baseline_accuracy':base[kind]['accuracy'],'accuracy':r[kind]['accuracy'],
                  'baseline_target_nll':base[kind]['target_nll'],'target_nll':r[kind]['target_nll']}
                for metric in ['target_nll','accuracy','first_nll','advantage','content_attention']:
                    values=np.array([y[metric]-x[metric] for x,y in zip(a,b)])
                    if metric=='accuracy':values*=100
                    result[kind][metric+'_delta']=paired(values,ix)
                result['by_length'][kind]=[]
                for n in sorted({v['n'] for v in a}):
                    ids=[i for i,x in enumerate(a) if x['n']==n]
                    result['by_length'][kind].append(dict(n=n,baseline_accuracy=float(np.mean([a[i]['accuracy'] for i in ids])),
                      accuracy=float(np.mean([b[i]['accuracy'] for i in ids])),nll_delta=float(np.mean([b[i]['target_nll']-a[i]['target_nll'] for i in ids]))))
            a=base['prose']['passages'];b=r['prose']['passages'];weights=[v['tokens'] for v in a]
            result['prose']=dict(nll_delta=paired([y['nll']-x['nll'] for x,y in zip(a,b)],pidx,weights),
              kl_from_baseline=r['prose']['kl_from_baseline'],argmax_changed=r['prose']['argmax_changed'],tokens=r['prose']['tokens'])
            ans.append(result)
    return ans

def norm(v,g):
    v=np.asarray(v,np.float64)
    return float(np.sqrt(max(0.,np.mean(np.einsum('bi,ij,bj->b',v,g,v,optimize=True)))))

def interactions():
    triples=read('sibling_triples.json');selection=set(read('selection_locked.json')['candidate_ids']);rows=[]
    for phase,step in [('rollback',1000),('transplant',512)]:
        gram=np.load(OUT/f'logit_gram_step{step}.npy')
        base=np.load(RAW/f'confirmation_{phase}_baseline.npz')
        for tr in triples:
            if not {tr['parent'],tr['left'],tr['right']}.issubset(selection):continue
            files=[np.load(RAW/f'confirmation_{phase}_{tr[k]}.npz') for k in ['parent','left','right']]
            for kind in ['repeat','scrambled']:
                key=kind+'_hidden';effects=[f[key].astype(np.float64)-base[key] for f in files]
                parent,a,b=effects;res=parent-a-b;den=norm(a,gram)+norm(b,gram)
                rows.append(dict(phase=phase,probe=kind,**tr,parent_logit_rms=norm(parent,gram),child_a_logit_rms=norm(a,gram),child_b_logit_rms=norm(b,gram),
                  interaction_logit_rms=norm(res,gram),interaction_fraction=norm(res,gram)/(den+1e-30)))
            for f in files:f.close()
        base.close()
    return rows

def native_metadata():
    source=np.load(ROOT/'native_qkv_curves.npz');delta=[source[f'step1000_layer{l}'].astype(np.float64)-source[f'step512_layer{l}'] for l in range(6)];meta=[]
    for c in read('candidate_catalog.json'):
        v=project(c,delta);m=dict(**c,update_norm=float(np.linalg.norm(v)),maximum_absolute_weight_update=float(abs(v).max()),nonzero_native_entries=int(np.count_nonzero(v)))
        if c['basis']=='packet':m['coefficient_count']=(c['end']-c['start'])//(2**len(c['path']));m['native_support_count']=c['end']-c['start']
        elif c['basis']=='haar':m['coefficient_count']=1;m['native_support_count']=2**c['level']
        else:m['complex_bin_count']=c['high']-c['low'];m['native_support_count']=v.size
        meta.append(m)
    return meta

def main():
    con=read('confirmation.json');later=read('later_context.json')
    effects=summarize_cases(con+later);dump('confirmation_effects.json',effects)
    interactions_rows=interactions();dump('behavioral_interactions.json',interactions_rows)
    meta=native_metadata();dump('component_metadata.json',meta)
    sel=read('selection_locked.json');leaf=sel['strongest_local_leaf']
    chosen=[r for r in effects if r['id'] in ['H6_root',leaf,'L3_fourier_atom','L3_haar_j6','H6_query','H6_key','H5_root','H7_root','head6_zero']]
    baseline=[dict(phase=r['phase'],repeat_accuracy=r['repeat']['accuracy'],scrambled_accuracy=r['scrambled']['accuracy'],prose_nll=r['prose']['nll']) for r in con+later if r['id']=='baseline']
    summary=dict(baselines=baseline,selected_leaf=leaf,selected_leaf_metadata=next(c for c in meta if c['id']==leaf),primary_results=chosen,
      discovery=read('discovery_environment.json'),confirmation=read('confirmation_environment.json'),
      total_evaluated_candidate_variants=len(read('discovery.json'))-1+len(con)-2+len(later)-1,
      bootstrap='4000 paired resamples, stratified by prefix length; 48 independent prefix sequences. Prose resampling over 8 passages, token-weighted mean.',
      semantic_scope='Causal contribution to synthetic next-token copying/content-following. No natural-language concept identity is established.',
      transfer_scope='One training seed; inference-only checkpoint hybrids. Small prose controls cannot establish preservation of general language ability.',
      hierarchy_scope='The selected depth-six packet contains 1024 coefficients over a 65536-weight native window; this is a full-resolution component group, not one coefficient or 1024 modified weights.',
      numerical_scope='Float64 projections from original checkpoint parameters, rounded once into float32 inference; exact restores; final-query full-vocabulary centered logit interaction is reconstructed from hidden states and the fixed output-head Gram matrix.')
    dump('summary.json',summary)
    print(json.dumps(dict(baselines=baseline,leaf=leaf,results=[dict(id=r['id'],phase=r['phase'],accuracy=r['repeat']['accuracy'],nll_delta=r['repeat']['target_nll_delta'],prose=r['prose']['nll_delta']) for r in chosen]),indent=2))

if __name__=='__main__':main()
