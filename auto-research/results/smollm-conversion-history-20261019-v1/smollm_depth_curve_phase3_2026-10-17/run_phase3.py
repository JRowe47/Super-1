"""Pilot, distill, then test a learned first-block weight curve at variable depth."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, platform, resource, time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from prepare_data import model_dir
from depth_curve import DepthCurveMLP, install_student, make_from_state, accounting, spline_basis
from distill import teacher_pairs, initialize, local_metrics, train_one
from evaluation import call_block, reference_cache, evaluate, digest_tensor, cluster_ci, paired_timing
from phase2_loader import load_first_block

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
SPECS=[{'id':'direct_192','regime':'fixed','width':192,'knots':1,'steps':1},
       {'id':'direct_768','regime':'fixed','width':768,'knots':1,'steps':1},
       {'id':'curve_fixed8','regime':'fixed','width':192,'knots':4,'steps':8},
       {'id':'curve_mixed','regime':'mixed','width':192,'knots':4,'steps':8}]
QK_SPEC={'id':'joint_activation_svd_0.125','method':'joint_activation_svd','budget':.125}


def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def load_student(spec):
    info=json.loads((OUT/f"training_{spec['id']}.json").read_text())
    return make_from_state(info['configuration'],load_file(str(ROOT/'checkpoints'/f"{spec['id']}.safetensors"))).eval()


def prepare(model,source,windows):
    train=teacher_pairs(model,source,windows['train'],'train')
    val=teacher_pairs(model,source,windows['validation'],'validation')
    initials={};init_meta=[]
    for width,knots in [(192,1),(192,4),(768,1)]:
        name=f'initial_w{width}_k{knots}';path=ROOT/'checkpoints'/f'{name}.safetensors';meta=ROOT/'checkpoints'/f'{name}.json'
        if path.exists():
            info=json.loads(meta.read_text());student=make_from_state(info['configuration'],load_file(str(path)))
        else:
            student,info=initialize(source,train,width,knots);info['configuration']=student.configuration()
            save_file(student.state_dict(),str(path));dump(meta,info)
        initials[(width,knots)]=student;init_meta.append(info)
    dump(OUT/'initialization.json',init_meta)
    return train,val,initials


@torch.no_grad()
def exact_control(model,source,windows):
    student=DepthCurveMLP(576,1536,1,source.post_attention_layernorm.weight,source.post_attention_layernorm.variance_epsilon)
    flat=torch.cat([source.mlp.gate_proj.weight.flatten(),source.mlp.up_proj.weight.flatten(),source.mlp.down_proj.weight.flatten()])
    student.coefficients[0].copy_(flat)
    ids=windows['validation'][:2,:-1];model.model.layers[0]=source
    original=model(ids,use_cache=False).logits
    install_student(model,source,student,1)
    changed=model(ids,use_cache=False).logits
    difference=float((original-changed).abs().max())
    model.model.layers[0]=source
    assert difference==0,difference
    return {'full_width_one_step_max_logit_difference':difference,'tested_tokens':ids.numel()}


def run_training(model,source,windows,updates,stage):
    train,val,initials=prepare(model,source,windows)
    controls=exact_control(model,source,windows);dump(OUT/'exact_control.json',controls)
    pilot_path=OUT/'pilot.json'
    if not pilot_path.exists():
        _,pilot=train_one(initials[(192,4)],train,val,{'id':'pilot','regime':'fixed','steps':8},updates=40,pilot=True)
        pilot['excluded_from_main']=True;pilot['purpose']='Check finite gradients, initial learning and wall time; main training restarts from original initialization.'
        dump(pilot_path,pilot)
        print('PILOT',json.dumps({k:pilot[k] for k in ['elapsed_seconds','best_validation_relative_mse','best_update']}),flush=True)
    if stage=='pilot':return
    protocol={'scope':'Only block-0 MLP is distilled; original attention executes once; all other model tensors frozen.',
       'depth_equation':'u[0]=0; u[i+1]=u[i]+F(a+u[i], i/N)/N; output=a+u[N]. F is RMSNorm plus SwiGLU with depth-dependent weights.',
       'curves':'Four clamped Catmull-Rom control vectors hold all gate/up/down weights. Continuous C1 weight curve; same controls at all evaluated depths.',
       'train_specs':SPECS,'updates_per_student':updates,'batch_tokens':512,'seed':30317,
       'objective':'Mean-squared error of first-block MLP residual output, divided by mean training target squared magnitude. No gradients through later blocks.',
       'optimizer':{'name':'AdamW','peak_lr':.0003,'weight_decay':.01,'warmup_updates':40,'cosine_floor':.1,'gradient_clip_norm':1.},
       'checkpoint_selection':'Every 100 updates plus initialization/final: minimum validation endpoint MSE. Fixed8 uses N=8; mixed uses equal mean across N=1,2,4,8; direct uses N=1.',
       'validation_gate':{'delta_nll_max':.05,'kl_max':.05},
       'test_candidates':'All four depths for both trained curve students, both direct students, fixed phase-2 Q/K baseline, and validation-selected curve composed with that Q/K baseline.',
       'curve_selection':'Minimum field evaluations passing both validation gates, then minimum KL; if no curve passes, minimum KL is explicitly a failed-gate representative.',
       'precision':'float32','device':'CPU','threads':4,'full_model_evaluation_batch':2,
       'data_manifest_sha256':hashlib.sha256((ROOT/'data/dataset_manifest.json').read_bytes()).hexdigest(),
       'architecture_limit':'Only the MLP sub-transformation has variable depth. No removal of the 30 outer blocks, no attention-cache compression, no adaptive solver or input-conditioned spline basis.'}
    dump(OUT/'protocol.json',protocol)
    all_training=[]
    for spec in SPECS:
        summary_path=OUT/f"training_complete_{spec['id']}.json"
        if summary_path.exists():all_training.append(json.loads(summary_path.read_text()));continue
        _,summary=train_one(initials[(spec['width'],spec['knots'])],train,val,spec,updates=updates)
        dump(summary_path,summary);all_training.append(summary)
    dump(OUT/'training_summary.json',all_training)


def configurations():
    rows=[{'id':'phase2_shared_qk','kind':'qk'}]
    for spec in SPECS:
        depths=[1,2,4,8] if spec['knots']>1 else [1]
        rows.extend({'id':f"{spec['id']}_n{n}",'kind':'student','student':spec['id'],'steps':n} for n in depths)
    return rows


def set_candidate(model,source,students,config,materialized=False):
    model.model.layers[0]=copy.deepcopy(source)
    if config['kind']=='qk':
        load_first_block(model,ROOT/'reference/joint_activation_svd_0.125.safetensors',QK_SPEC)
        return {'block_coefficient_values':3152832,'block_projection_macs_per_token':3151680,
                'block_projection_mac_fraction_saved':387264/3538944,'field_evaluations':1,'sampled_matrix_cache_values':0}
    base=source
    if config['kind']=='combined':
        load_first_block(model,ROOT/'reference/joint_activation_svd_0.125.safetensors',QK_SPEC)
        base=model.model.layers[0]
    student=students[config['student']]
    install_student(model,base,student,config['steps'],materialized=materialized)
    return accounting(student,config['steps'],config['kind']=='combined',materialized)


@torch.no_grad()
def mechanism_diagnostics(student,anchor,target):
    """One prespecified validation window; exploratory, never used for selection."""
    results={};paths={}
    for n in [1,2,4,8,16,32,64]:
        delta,path=student(anchor,steps=n,return_path=True)
        paths[str(n)]=path.cpu().numpy()
        results[str(n)]={'relative_rms_error':float(((delta-target).square().sum()/target.square().sum()).sqrt())}
    frozen=student(anchor,steps=8,frozen_state=True)
    results['frozen_state_n8']={'relative_rms_error':float(((frozen-target).square().sum()/target.square().sum()).sqrt()),
          'difference_from_recurrent_relative_rms':float(((frozen-(torch.from_numpy(paths['8'][-1])-anchor)).square().sum()/target.square().sum()).sqrt())}
    c=student.coefficients.detach().double();mean=c.mean(0,keepdim=True)
    variation=float(((c-mean).square().sum()/c.square().sum()).sqrt())
    results['control_relative_variation']=variation
    gram=(c@c.T).cpu().numpy();times=np.linspace(0,1,129)
    bases=np.asarray([spline_basis(float(t),student.knots,dtype=torch.float64).numpy() for t in times])
    weight_norm=np.sqrt(np.einsum('ij,jk,ik->i',bases,gram,bases))
    db=np.gradient(bases,times,axis=0);ddb=np.gradient(db,times,axis=0)
    speed=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',db,gram,db)))
    bending=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',ddb,gram,ddb)))
    np.savez_compressed(OUT/'curve_trajectories.npz',**paths,anchor=anchor.numpy(),target=target.numpy(),
          depth=times,weight_frobenius_norm=weight_norm,weight_derivative_norm=speed,weight_second_derivative_norm=bending)
    dump(OUT/'mechanism_diagnostics.json',{'scope':'First validation window only; no selection uses this diagnostic.',
          'token_for_plot':127,'metrics':results,'note':'Weight-curve smoothness does not imply solver-resolution invariance. N=64 is a diagnostic, not a ground-truth path.'})


@torch.no_grad()
def run_evaluation(model,source,windows):
    students={s['id']:load_student(s) for s in SPECS}
    manifest=json.loads((ROOT/'data/dataset_manifest.json').read_text())
    model.model.layers[0]=copy.deepcopy(source);valref=reference_cache(model,windows['validation'],'validation')
    configs=configurations();valpath=OUT/'validation.json';rows=json.loads(valpath.read_text()) if valpath.exists() else []
    done={r['configuration']['id'] for r in rows}
    for config in configs:
        if config['id'] in done:continue
        cost=set_candidate(model,source,students,config);metrics=evaluate(model,windows['validation'],valref)
        assert metrics['all_finite'];rows.append({'configuration':config,'accounting':cost,'metrics':metrics});dump(valpath,rows)
        print('VALIDATION',config['id'],'dNLL',round(metrics['delta_nll'],6),'KL',round(metrics['mean_kl'],6),flush=True)
    curves=[r for r in rows if r['configuration'].get('student','').startswith('curve_')]
    passing=[r for r in curves if r['metrics']['delta_nll']<=.05 and r['metrics']['mean_kl']<=.05]
    if passing:selected=min(passing,key=lambda r:(r['configuration']['steps'],r['metrics']['mean_kl']))
    else:selected=min(curves,key=lambda r:r['metrics']['mean_kl'])
    chosen=selected['configuration'];combined={**chosen,'id':'selected_curve_plus_shared_qk','kind':'combined'}
    selection={'selection_split':'validation only','passed_gate':bool(passing),'selected_curve':chosen,'combined':combined,
               'reason':'Minimum steps passing delta NLL/KL gate, then KL.' if passing else 'No curve passed; lowest-KL failed-gate representative.'}
    dump(OUT/'selection.json',selection)
    # Lock specifications before constructing test caches or looking at test metrics.
    configs.append(combined)
    model.model.layers[0]=copy.deepcopy(source);testref=reference_cache(model,windows['test'],'test')
    path=OUT/'test.json';tested=json.loads(path.read_text()) if path.exists() else [];done={r['configuration']['id'] for r in tested}
    docids=[w['document_index'] for w in manifest['splits']['test']['selected_windows']]
    for config in configs:
        if config['id'] in done:continue
        cost=set_candidate(model,source,students,config);metrics=evaluate(model,windows['test'],testref)
        assert metrics['all_finite'];metrics['delta_nll_document_bootstrap_95ci']=cluster_ci(metrics['per_window_delta_nll'],docids)
        tested.append({'configuration':config,'accounting':cost,'metrics':metrics});dump(path,tested)
        print('TEST',config['id'],'dNLL',round(metrics['delta_nll'],6),'KL',round(metrics['mean_kl'],6),flush=True)
    timings=[]
    timing_configs=[c for c in configs if c.get('student')==chosen['student'] and c['kind']=='student']
    timing_configs+=[next(c for c in configs if c['id']=='direct_768_n1'),configs[0]]
    for config in timing_configs:
        for cache in ([False,True] if config.get('student')==chosen['student'] else [False]):
            cost=set_candidate(model,source,students,config,materialized=cache);candidate=model.model.layers[0]
            measured=paired_timing(model,source,candidate,windows['validation'][0:1,:-1])
            timings.append({'configuration':config,'materialized':cache,'accounting':cost,'measurements':measured})
            dump(OUT/'timing.json',timings)
        print('TIMING',config['id'],flush=True)
    # Cache compatibility and same-weight streamed/materialized equivalence.
    set_candidate(model,source,students,chosen)
    seq=windows['validation'][0:1,:32];full=model(seq,use_cache=False).logits
    cache=DynamicCache();incremental=[]
    for i in range(32):incremental.append(model(seq[:,i:i+1],past_key_values=cache,use_cache=True).logits)
    cache_diff=float((full-torch.cat(incremental,dim=1)).abs().max())
    set_candidate(model,source,students,chosen,materialized=True)
    materialized_diff=float((full-model(seq,use_cache=False).logits).abs().max())
    assert cache_diff<1e-3,cache_diff
    assert materialized_diff==0,materialized_diff
    valpairs=teacher_pairs(model,source,windows['validation'],'validation')
    mechanism_diagnostics(students[chosen['student']],valpairs['anchor'][:128],valpairs['target'][:128])
    tok=AutoTokenizer.from_pretrained(model_dir(),local_files_only=True)
    prompts=['The purpose of a scientific experiment is','The city council voted to','To solve the problem, we first']
    generations=[]
    for prompt in prompts:
        entry={'prompt':prompt};inputs=tok(prompt,return_tensors='pt')
        variants={'original':None,'curve_n8':{**chosen,'steps':8},'curve_n1':{**chosen,'steps':1}}
        for label,config in variants.items():
            if config is None:model.model.layers[0]=copy.deepcopy(source)
            else:set_candidate(model,source,students,config)
            out=model.generate(**inputs,max_new_tokens=40,do_sample=False,pad_token_id=tok.eos_token_id)
            entry[label]=tok.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        generations.append(entry)
    dump(OUT/'generations.json',generations)
    model.model.layers[0]=copy.deepcopy(source)
    return {'validation_candidates':len(rows),'test_candidates':len(tested),'test_reference_nll':float(testref['nll'].mean()),
       'selected_curve':chosen,'curve_passed_validation_gate':bool(passing),'cached_vs_prefill_max_logit':cache_diff,
       'materialized_vs_streamed_max_logit':materialized_diff,'test_document_clusters':len(set(docids))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','train','evaluate','all'],default='all');ap.add_argument('--updates',type=int,default=3000);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(30317);torch.use_deterministic_algorithms(True)
    os.environ['TOKENIZERS_PARALLELISM']='false';start=time.perf_counter()
    data=np.load(ROOT/'data/token_windows.npz');windows={k:torch.from_numpy(data[k]).long() for k in data.files}
    model=AutoModelForCausalLM.from_pretrained(model_dir(),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    model.requires_grad_(False);source=copy.deepcopy(model.model.layers[0])
    outside={n:digest_tensor(p) for n,p in model.named_parameters() if not n.startswith('model.layers.0.')}
    if args.stage in ['pilot','train','all']:
        run_training(model,source,windows,args.updates,args.stage)
        if args.stage in ['pilot','train']:return
    evaluated=run_evaluation(model,source,windows)
    after=dict(model.named_parameters());unchanged=all(digest_tensor(after[n])==h for n,h in outside.items());assert unchanged
    summary={'status':'PASS',**evaluated,'outside_parameter_tensors_checked':len(outside),'outside_parameters_identical':unchanged,
       'training_tokens':len(windows['train'])*128,'validation_tokens':len(windows['validation'])*128,'test_tokens':len(windows['test'])*128,
       'process_elapsed_seconds':time.perf_counter()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
       'torch':torch.__version__,'python':platform.python_version()}
    dump(OUT/'summary.json',summary);print('PHASE 3 COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
