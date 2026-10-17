"""Post-hoc mechanistic follow-up, registered before its interventions run.

The parent PCA20 fit badly damages block 0 and block 7. Compare restoring all
three original FFN matrices of block 0 with an equally sized restoration of the
most damaged other block, selected using parent TRAIN-window local errors only.
This is a diagnostic extension, not an independent confirmation or smooth curve.
"""
import argparse
import hashlib
import json
import time
import resource
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'tinyllama-depth-curves-20261017-v1'
REPO=ROOT.parents[2]
FAMILIES=['gate_proj','up_proj','down_proj']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def prepare():
    assert not (ROOT/'PROTOCOL.json').exists()
    parent=json.loads((PARENT/'FUNCTIONAL.json').read_text())
    reference=next(x for x in parent['rows'] if x['id']=='pca20')
    errors=reference['local_ffn_diagnostics']['train']['per_layer_relative_squared_output_error']
    other=int(np.argmax(np.asarray(errors)[1:])+1)
    protocol={
        'run_id':ROOT.name,'status':'FROZEN_BEFORE_INTERVENTION','parent_run':PARENT.name,
        'chronology':'Parent structural and all nine functional outcomes were observed. This is a post-hoc diagnosis, not part of the original gate or an independent validation.',
        'authorization':'User authorized exploring TinyLlama depth conversion; bounded local CPU follow-up within that exploratory scope. PR target JRowe47/Super-1 only.',
        'hypothesis':'PCA20 loses low-weight-energy but functionally critical early FFN computation. Restoring block0 should recover much more end-to-end quality than restoring the equally sized most locally damaged nonzero block.',
        'comparator':'Identical PCA20 base with one original FFN added: block0 vs nonzero block selected solely from parent TRAIN-window local relative output errors.',
        'other_block':other,'selection_train_errors':errors,
        'arms':[{'id':'teacher','restore':None},{'id':'pca20_replay','restore':None},
                {'id':f'pca20_restore_block{other}','restore':other},{'id':'pca20_restore_block0','restore':0}],
        'primary_metric':'Same 2048-target validation mean next-token CE, nats/token.',
        'useful_diagnostic_effect':'Block0 restoration improves CE at least 1.0 nat/token more than the matched other-block restoration.',
        'parameter_accounting':'Teacher unchanged. PCA20 stores 20 full matrices per family plus 3*22*19 free depth coefficients. Restoration retains all PCA bases/coefficients plus 3*5632*2048 original FFN scalars. Original block shapes are identical.',
        'data':'Exact same frozen parent windows and tokenizer; repeated validation is exploratory. No new validation-driven selection. Official test untouched.',
        'training_updates':0,'resources':'4 CPU threads, no GPU or paid compute, <=10 minutes execution',
        'fidelity_tolerances':{'teacher_ce_abs':2e-6,'pca20_replay_ce_abs':2e-6},
        'source_sha256':sha(__file__),'parent_protocol_sha256':sha(PARENT/'PROTOCOL.json'),
        'parent_functional_sha256':sha(PARENT/'FUNCTIONAL.json'),'parent_projectors_sha256':sha(PARENT/'PROJECTORS.json'),
        'parent_structure_sha256':sha(PARENT/'STRUCTURE.json'),
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write('PROTOCOL.json',protocol)
    write('REGISTRATION.json',{'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'status':'FROZEN_BEFORE_INTERVENTION'})
    path=REPO/'auto-research/RUN_REGISTRY.json'
    registry=json.loads(path.read_text())
    assert not any(r['run_id']==ROOT.name for r in registry['runs'])
    registry['runs'].append({'run_id':ROOT.name,'status':'FROZEN_BEFORE_INTERVENTION',
        'protocol':str((ROOT/'PROTOCOL.json').relative_to(REPO)),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),
        'metrics':str((ROOT/'RESULTS.json').relative_to(REPO)),
        'report':f'auto-research/reports/{ROOT.name}.md','publication_target':'JRowe47/Super-1 main only'})
    path.write_text(json.dumps(registry,indent=2)+'\n')
    print('FROZEN',sha(ROOT/'PROTOCOL.json'),'other block',other,flush=True)


def run(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert sha(__file__)==p['source_sha256']
    assert sha(PARENT/'FUNCTIONAL.json')==p['parent_functional_sha256']
    assert sha(PARENT/'PROJECTORS.json')==p['parent_projectors_sha256']
    assert sha(ROOT/'PROTOCOL.json')==json.loads((ROOT/'REGISTRATION.json').read_text())['protocol_sha256']
    parent=json.loads((PARENT/'FUNCTIONAL.json').read_text())
    parent_rows={x['id']:x for x in parent['rows']}
    projectors=json.loads((PARENT/'PROJECTORS.json').read_text())
    shapes=json.loads((PARENT/'STRUCTURE.json').read_text())
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    started,cpu=time.monotonic(),time.process_time()
    model=AutoModelForCausalLM.from_pretrained(args.inputs/'model',local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa',use_safetensors=True).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    windows=np.load(args.cache/'windows.npz')['validation']
    assert sha(args.cache/'windows.npz')==json.loads((PARENT/'REGISTRATION.json').read_text())['windows_npz_sha256']
    nparams=sum(t.numel() for t in model.parameters())
    per_ffn=sum(shapes[f]['scalars_per_matrix'] for f in FAMILIES)
    teacher_logp=[];rows=[]
    with safe_open(args.inputs/'model/model.safetensors',framework='pt',device='cpu') as source:
        for arm in p['arms']:
            arm_start,arm_cpu=time.monotonic(),time.process_time()
            print('START',arm['id'],flush=True)
            if arm['id']!='teacher':
                for family in FAMILIES:
                    mix=np.array(projectors[f'{family}_pca20'])
                    if arm['restore'] is not None:
                        mix[arm['restore'],:]=0.;mix[arm['restore'],arm['restore']]=1.
                    mix=torch.tensor(mix,dtype=torch.float32)
                    src=[source.get_tensor(f'model.layers.{i}.mlp.{family}.weight').reshape(-1) for i in range(22)]
                    n=src[0].numel();fitted=torch.empty((22,n),dtype=torch.float32)
                    with torch.inference_mode():
                        for offset in range(0,n,262144):
                            x=torch.stack([w[offset:offset+262144] for w in src])
                            torch.mm(mix,x,out=fitted[:,offset:offset+x.shape[1]])
                    for i,layer in enumerate(model.model.layers):
                        getattr(layer.mlp,family).weight=torch.nn.Parameter(fitted[i].reshape(shapes[family]['shape']),requires_grad=False)
                    del fitted,src
            ce=[];kl=[];agreement=[]
            with torch.inference_mode():
                for index,window in enumerate(windows):
                    logits=model(torch.as_tensor(window[:-1])[None,:],use_cache=False).logits[0]
                    assert torch.isfinite(logits).all()
                    lp=F.log_softmax(logits,dim=-1)
                    ce.append(float(F.nll_loss(lp,torch.as_tensor(window[1:]))))
                    if arm['id']=='teacher':teacher_logp.append(lp.clone())
                    t=teacher_logp[index]
                    kl.append(float((t.exp()*(t-lp)).sum(-1).mean()))
                    agreement.append(float((t.argmax(-1)==lp.argmax(-1)).float().mean()))
            stored=nparams if arm['id']=='teacher' else nparams-22*per_ffn+20*per_ffn+3*22*19
            if arm['restore'] is not None:stored+=per_ffn
            row={**arm,'validation_ce':float(np.mean(ce)),'perplexity':float(np.exp(np.mean(ce))),
                 'forward_kl':float(np.mean(kl)),'teacher_top1_agreement':float(np.mean(agreement)),
                 'window_ce':ce,'window_kl':kl,'window_agreement':agreement,
                 'analytical_stored_scalars':stored,'analytical_weight_saving_fraction':1-stored/nparams,
                 'wall_seconds':time.monotonic()-arm_start,'cpu_seconds':time.process_time()-arm_cpu}
            if arm['id'] in ['teacher','pca20_replay']:
                prior=parent_rows['teacher' if arm['id']=='teacher' else 'pca20']
                row['max_window_ce_replay_abs']=float(np.max(np.abs(np.array(ce)-np.array(prior['window_ce']))))
                assert row['max_window_ce_replay_abs']<=2e-6
            rows.append(row)
            write('RESULTS.json',{'rows':rows,'targets_per_arm':2048,'post_hoc_diagnostic':True})
            print('COMPLETE',arm['id'],row['validation_ce'],row['forward_kl'],row['teacher_top1_agreement'],flush=True)
    write('TERMINAL.json',{'status':'COMPLETED','completed_arms':len(rows),'declared_arms':len(p['arms']),
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'official_test_accessed':False,'training_updates':0,'cloud_resources_provisioned':False,'followup_running':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:run(args)
