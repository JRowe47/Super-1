"""Lock validation selection, evaluate held-out test, verify caches and exports."""
import gc,json,math,platform,resource,statistics,time
import numpy as np
import torch
from safetensors.torch import load_file
from common import ROOT,OUT,WINDOWS,dump,setup,load_model,hashes,data
from curve_model import HermiteBank,CurveBlock,install,restore,window_forward
from train import load_bank
from prepare import cache_split,cache_check
from evaluation import evaluate,cluster_ci


def configurations():
    rows=[]
    for start in WINDOWS:
        for kind,depths in [('initial',[3,2]),('trained',[4,3,2]),('drop',[3,2])]:
            for n in depths:
                rows.append({'id':f'w{start}_{kind}_n{n}','windows':[{'start':start,'kind':kind,'steps':n}]})
    return rows


def specs_for(config,banks):
    specs={}
    for item in config['windows']:
        s,n,kind=item['start'],item['steps'],item['kind']
        specs[s]={'keep':[0,1,3] if n==3 else [0,3]} if kind=='drop' else {'bank':banks[(s,kind)],'steps':n}
    return specs


def accounting(config):
    removed=sum(4-i['steps'] for i in config['windows'])
    learned=sum(i['kind']=='trained' for i in config['windows'])
    active=30-removed
    # Matrix MACs for all seven decoder projections; attention score/value
    # matmuls, normalizations and curve sampling excluded and stated explicitly.
    return {'decoder_evaluations':active,'removed_blocks':removed,
        'decoder_projection_macs_per_token':active*3538944,
        'decoder_projection_mac_fraction_saved':removed/30,
        'added_trained_slope_values':learned*14160384,
        'added_trained_slope_bytes_fp32':learned*14160384*4,
        'materialized_curve_matrix_bytes_fp32':sum(i['steps']*3540096*4 for i in config['windows'] if i['kind']!='drop'),
        'native_original_unique_values_retained':134515008,
        'note':'Retains original weight knots and adds slopes; no parameter compression. MAC counts exclude head, attention products and curve sampling; measured latency is reported separately.'}


@torch.no_grad()
def timing(model,original,config,banks,ids):
    specs=specs_for(config,banks);result={}
    for materialized in [False,True]:
        pairs=[]
        for repeat in range(14):
            row={}
            for mode in (['native','curve'] if repeat%2==0 else ['curve','native']):
                if mode=='native':restore(model,original)
                else:install(model,original,specs,materialized=materialized)
                start=time.perf_counter();model(ids,use_cache=False).logits
                row[mode]=(time.perf_counter()-start)*1000
            if repeat>=2:pairs.append(row)
        result['materialized' if materialized else 'streamed']={
            'native_median_ms':statistics.median(p['native'] for p in pairs),
            'curve_median_ms':statistics.median(p['curve'] for p in pairs),
            'median_paired_ratio':statistics.median(p['curve']/p['native'] for p in pairs),
            'pairs_ms':pairs,'note':'Single-process CPU, 4 threads, batch 1, context 128. Installation/pre-materialization outside timed region; forward and output logits included.'}
    restore(model,original);return result


@torch.no_grad()
def slope_diagnostics(banks):
    result={}
    for start in WINDOWS:
        b=banks[(start,'trained')];init=banks[(start,'initial')]
        per_family={}
        for name in b.slopes:
            x=b.slopes[name].double();y=init.slopes[name].double()
            per_family[name]={'initial_rms':float(y.square().mean().sqrt()),
                'trained_rms':float(x.square().mean().sqrt()),
                'relative_change':float(((x-y).square().sum()/y.square().sum()).sqrt())}
        # Gram matrices compute exact full-dimensional curve norms without
        # sampling or plotting millions of individual weight coordinates.
        points=[]
        for j in range(3):
            gram=torch.zeros((4,4),dtype=torch.float64)
            for name in b.names:
                sk=name.replace('.','__')
                coeff=torch.stack([b.knots[j][name].flatten(),b.slopes[sk][j].detach().flatten(),
                                   b.knots[j+1][name].flatten(),b.slopes[sk][j+1].detach().flatten()]).double()
                gram+=coeff@coeff.T
            for u in np.linspace(0,1,33,endpoint=j==2):
                h=torch.tensor([2*u**3-3*u**2+1,u**3-2*u**2+u,-2*u**3+3*u**2,u**3-u**2],dtype=torch.float64)
                d=torch.tensor([6*u**2-6*u,3*u**2-4*u+1,-6*u**2+6*u,3*u**2-2*u],dtype=torch.float64)
                points.append({'depth':start+j+float(u),'weight_norm':float((h@gram@h).clamp_min(0).sqrt()),
                    'slope_norm':float((d@gram@d).clamp_min(0).sqrt())})
        result[str(start)]={'families':per_family,'profile':points}
    dump(OUT/'curve_diagnostics.json',result)


@torch.no_grad()
def main():
    setup();windows=data();model=load_model();original=list(model.model.layers);before=hashes(model)
    banks={}
    for s in WINDOWS:
        for kind in ['initial','trained']:
            banks[(s,kind)]=HermiteBank(original[s:s+4])
            if kind=='trained':load_bank(banks[(s,kind)],ROOT/'checkpoints'/f'window_{s}.safetensors')
    configs=configurations();valref=torch.load(ROOT/'cache/validation.pt',weights_only=True)
    # Only endpoint reference hidden states needed during full-model scoring.
    valref={k:valref[k] for k in ['hidden','nll']}
    valpath=OUT/'validation.json';rows=json.loads(valpath.read_text()) if valpath.exists() else []
    done={r['configuration']['id'] for r in rows}
    for config in configs:
        if config['id'] in done:continue
        install(model,original,specs_for(config,banks));metrics=evaluate(model,windows['validation'],valref)
        assert metrics['all_finite']
        if config['windows'][0]['steps']==4:assert metrics['max_logit_abs']==0
        rows.append({'configuration':config,'accounting':accounting(config),'metrics':metrics});dump(valpath,rows)
        print('VALIDATION',config['id'],round(metrics['delta_nll'],5),round(metrics['mean_kl'],5),flush=True)
    selected=[];selection=[]
    for s in WINDOWS:
        candidates=[r for r in rows if len(r['configuration']['windows'])==1 and r['configuration']['windows'][0]['start']==s and r['configuration']['windows'][0]['kind']=='trained' and r['configuration']['windows'][0]['steps']<4]
        passing=[r for r in candidates if r['metrics']['delta_nll']<=.05 and r['metrics']['mean_kl']<=.05]
        chosen=min(passing,key=lambda r:(r['configuration']['windows'][0]['steps'],r['metrics']['mean_kl'])) if passing else min(candidates,key=lambda r:r['metrics']['mean_kl'])
        selected+=chosen['configuration']['windows']
        selection.append({'start':s,'selected':chosen['configuration']['id'],'passed_gate':bool(passing)})
    composite={'id':'selected_composite','windows':selected}
    dump(OUT/'selection.json',{'per_window':selection,'composite':composite,'selection_split':'validation only','test_not_consulted':True})
    configs.append(composite)
    install(model,original,specs_for(composite,banks));metrics=evaluate(model,windows['validation'],valref)
    rows=[r for r in rows if r['configuration']['id']!='selected_composite']
    rows.append({'configuration':composite,'accounting':accounting(composite),'metrics':metrics});dump(valpath,rows)
    restore(model,original);cache_split(model,windows['test'],'test',False)
    testref=torch.load(ROOT/'cache/test.pt',weights_only=True)
    manifest=json.loads((ROOT/'data/dataset_manifest.json').read_text());docids=[x['document_index'] for x in manifest['splits']['test']]
    testpath=OUT/'test.json';tested=json.loads(testpath.read_text()) if testpath.exists() else [];done={r['configuration']['id'] for r in tested}
    for config in configs:
        if config['id'] in done:continue
        install(model,original,specs_for(config,banks));metrics=evaluate(model,windows['test'],testref)
        assert metrics['all_finite']
        if len(config['windows'])==1 and config['windows'][0]['steps']==4:assert metrics['max_logit_abs']==0
        metrics['delta_nll_document_bootstrap_95']=cluster_ci(metrics['per_window_delta_nll'],docids)
        tested.append({'configuration':config,'accounting':accounting(config),'metrics':metrics});dump(testpath,tested)
        print('TEST',config['id'],round(metrics['perplexity_ratio_to_original'],5),round(metrics['mean_kl'],5),flush=True)
    checks={}
    for config in configs:
        if config['id']!='selected_composite' and not ('trained' in config['id'] and config['windows'][0]['steps']<4):continue
        install(model,original,specs_for(config,banks));check=cache_check(model,windows['test'][:1,:32]);checks[config['id']]=check
        print('CACHE_CHECK',config['id'],check,flush=True)
    dump(OUT/'cache_checks.json',checks)
    # Reconstruct each curve from checkpoint + original knots and check identical
    # sampled weights/outputs; original native parameters must remain untouched.
    export={}
    for s in WINDOWS:
        fresh=HermiteBank(original[s:s+4]);load_bank(fresh,ROOT/'checkpoints'/f'window_{s}.safetensors')
        config={'windows':[{'start':s,'steps':2,'kind':'trained'}]}
        install(model,original,{s:{'bank':banks[(s,'trained')],'steps':2}});a=model(windows['test'][:1,:32],use_cache=False).logits
        install(model,original,{s:{'bank':fresh,'steps':2}});b=model(windows['test'][:1,:32],use_cache=False).logits
        export[str(s)]=float((a-b).abs().max());assert export[str(s)]==0
        del fresh
    restore(model,original);assert before==hashes(model)
    dump(OUT/'export_verification.json',{'reloaded_curve_max_logit_difference':export,'original_parameter_hashes_unchanged':True})
    times={}
    for config in [next(c for c in configs if c['id']==r['selected']) for r in selection]+[composite]:
        times[config['id']]=timing(model,original,config,banks,windows['test'][:1,:-1])
        print('TIMING',config['id'],{k:v['median_paired_ratio'] for k,v in times[config['id']].items()},flush=True)
    dump(OUT/'timing.json',times);slope_diagnostics(banks)
    dump(OUT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),
        'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'device':'CPU','precision':'float32'})
    print('EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
