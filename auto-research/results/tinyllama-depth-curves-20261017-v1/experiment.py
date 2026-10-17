"""A bounded, training-free screen of depth structure in pretrained TinyLlama.

Compare centered PCA with a constant mode, cubic B-splines, and low-frequency
cosines at matched counts of full matrix bases. Reconstruction is along depth,
never across arbitrary within-matrix element ordering. All 22 blocks execute.
"""
import argparse
import hashlib
import inspect
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FAMILIES = ['gate_proj', 'up_proj', 'down_proj']
RANKS = [4, 8, 12, 16, 20, 22]
SEED = 101701


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    path = ROOT / name
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def event(kind, **kwargs):
    obj = {'event': kind, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs}
    with (ROOT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')
    print(json.dumps(obj), flush=True)


def depth_basis(kind, count, gram):
    n = len(gram)
    if kind == 'pca':
        c = np.eye(n) - np.ones((n, n)) / n
        values, vectors = np.linalg.eigh(c @ gram @ c)
        raw = np.column_stack([np.ones(n) / np.sqrt(n), vectors[:, np.argsort(values)[::-1][:count-1]]])
    elif kind == 'spline':
        knots = np.r_[np.zeros(4), np.arange(1, count-3) / (count-3), np.ones(4)]
        raw = BSpline.design_matrix(np.linspace(0, 1, n), knots, 3).toarray()
    elif kind == 'cosine':
        raw = np.cos(np.pi * (np.arange(n)[:, None] + 0.5) * np.arange(count)[None, :] / n)
    else:
        raise ValueError(kind)
    if np.linalg.matrix_rank(raw) != count:
        raise ValueError(f'Rank-deficient {kind} basis')
    q, _ = np.linalg.qr(raw, mode='reduced')
    return q, q @ q.T


def prepare(args):
    import torch
    import transformers
    import safetensors
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from transformers.models.llama import modeling_llama
    if (ROOT / 'PROTOCOL.json').exists():
        raise RuntimeError('Immutable protocol already exists')
    inputs = json.loads((ROOT / 'INPUTS.json').read_text())
    expected = {Path(x['path']).name: x for x in inputs['files']}
    for split in ['train', 'validation']:
        assert sha(args.inputs / f'{split}.parquet') == expected[f'{split}.parquet']['sha256']
    tokenizer = AutoTokenizer.from_pretrained(args.inputs / 'model', local_files_only=True)
    # Join non-empty raw lines in their published order. No official test access.
    windows = {}
    data_summary = {}
    for split, count in [('train', 4), ('validation', 16)]:
        lines = pq.read_table(args.inputs / f'{split}.parquet', columns=['text'])['text'].to_pylist()
        text = '\n\n'.join(x for x in lines if x.strip())
        ids = np.asarray(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'], dtype=np.int64)
        starts = np.linspace(0, len(ids) - 129, count, dtype=np.int64)
        windows[split] = np.stack([ids[s:s+129] for s in starts])
        data_summary[split] = {'full_token_count': len(ids), 'starts': starts.tolist(), 'windows': count,
                               'inputs_per_window': 128, 'targets_per_window': 128,
                               'full_token_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                               'window_ids_sha256': hashlib.sha256(windows[split].tobytes()).hexdigest()}
    args.cache.mkdir(parents=True, exist_ok=True)
    np.savez(args.cache / 'windows.npz', **windows)
    # Publish only token positions and hashes, not copies of corpus text.
    arms = [{'id': 'teacher', 'kind': 'original', 'rank': 22},
            {'id': 'spline22_roundtrip', 'kind': 'spline', 'rank': 22}]
    candidates = [{'id': f'{kind}{k}', 'kind': kind, 'rank': k}
                  for k in [8, 16, 20] for kind in ['pca', 'spline']]
    candidates += [{'id': 'cosine8', 'kind': 'cosine', 'rank': 8}]
    rng = np.random.default_rng(SEED)
    rng.shuffle(candidates)
    arms += candidates
    config = json.loads((args.inputs / 'model/config.json').read_text())
    assert (config['num_hidden_layers'], config['hidden_size'], config['intermediate_size']) == (22, 2048, 5632)
    template = json.loads((REPO / 'auto-research/PROTOCOL_TEMPLATE.json').read_text())
    template.update(status='FROZEN_BEFORE_MEASUREMENT', experiment_id=ROOT.name, run_id=ROOT.name,
        parent_run_ids=[], hypothesis='Corresponding pretrained FFN matrices have a small shared depth subspace, and a smooth depth curve retains it without recovery training.',
        primary_comparator='Unchanged pretrained TinyLlama; centered PCA plus constant is the non-smooth matched-matrix-count control.',
        primary_metric='Validation mean next-token cross-entropy in nats per TinyLlama token at 128-token reset context; secondary full-vocabulary forward KL and teacher top-1 agreement.',
        minimum_useful_effect='At K=8, <=0.10 nats/token CE increase vs teacher and <=0.05 nats/token spline disadvantage vs PCA. These are exploratory continuation thresholds, not statistical confirmation.',
        success_rule='Both K=8 thresholds must hold. Structure-only success cannot satisfy the functional gate.',
        futility_rule='Stop on failed identity/finite checks, more than 30 minutes execution, or memory pressure. Retain every declared outcome; no selection by validation to omit failed arms.')
    template['authorization'] = dict(approved=True,
        user_approval_reference='2026-10-16 America/Denver: "go ahead with the experiments, lets explore" following the proposed TinyLlama FFN-only depth-curve comparison; prior instruction directs PRs to JRowe47/Super-1.',
        approved_resources='Bounded local CPU exploratory measurements using existing workspace; no paid compute, GPU, large training, or official test.',
        max_cost='No paid services', max_walltime='30 minutes for measurement, excluding download/setup/publication', local_cpu_budget='4 threads, at most 120 aggregate CPU minutes')
    template['design'] = dict(original_source_hashes={str(Path(modeling_llama.__file__).name):sha(modeling_llama.__file__)},
        mechanism_equations='For each FFN tensor family X[L,P], Gram G=X X^T; fitted weights Xhat=Q Q^T X. PCA: constant + top K-1 eigenvectors of centered G. Spline: uniform clamped cubic B-spline column space, K controls. Cosine: first K DCT-II modes. Evaluate full pretrained model after replacing only FFN weights.',
        arms=arms, structure_ranks=RANKS, families=FAMILIES,
        control_rationale='Same K full dense matrices per tensor role; PCA coefficients unrestricted across layers, spline/cosine depth spaces fixed. Rank22 roundtrip validates implementation. 64 shuffled-depth controls assess natural-order structure without model retraining.',
        seeds=[SEED], seed_selection_method='Fixed seed for arm order and depth permutations; one fixed pretrained checkpoint, no training seeds.',
        learning_rate_trials_per_arm=0, parameter_counts='Computed exactly from checkpoint shapes. K*3*5632*2048 basis scalars, plus PCA coefficient metadata and unchanged non-FFN parameters.',
        matching_track='Full matrix basis storage count, not FLOP or quality matching; reconstruction is materialized for functional checks.',
        compute_accounting='Per-stage and per-arm wall/process CPU time; maximum process RSS. These timings are not deployment-speed benchmarks.')
    template['data'] = dict(dataset_versions={'model':inputs['model'], 'model_revision':inputs['model_revision'], 'dataset':inputs['dataset'], 'dataset_revision':inputs['dataset_revision']},
        tokenizer_sha256=expected['tokenizer.json']['sha256'],train_sha256=expected['train.parquet']['sha256'],selection_sha256=expected['validation.parquet']['sha256'],
        confirmation_sha256='Not applicable: exploratory validation only; no independent confirmation dataset',
        split_boundaries=data_summary, research_wide_region_reuse_audit='Official validation has been used in earlier project work; prior tokenizer and sample boundaries differ. Not an untouched confirmation set. Pretraining contamination not audited.',
        teacher_and_objective='Public TinyLlama 1.1B 3T base checkpoint; no OpenAI model outputs used as training data; no gradient training in this pass.', official_test_access_permitted=False)
    template['training'] = dict(updates=0,target_exposures_per_trial=0,tokens_per_update='Not applicable',microbatch='Not applicable',gradient_accumulation='Not applicable',
        bptt_length='Not applicable',accessible_history='At most 128 current-window inputs',state_carry_and_detach='KV cache disabled; reset each window',
        document_reset_policy='Fixed windows on concatenated split lines; may cross article boundaries; no cross-window state',optimizer_schedule='No optimizer; exact weight-space least squares',
        precision='FP64 depth Gram/basis; FP32 reconstruction and model evaluation',checkpoint_selection_rule='All declared arms, no best-checkpoint selection')
    template['execution'] = dict(hardware={'platform':platform.platform(),'logical_cpus':os.cpu_count(),'threads':4,'gpu_used':False},
        runtime_versions={'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'safetensors':safetensors.__version__},
        resolved_writable_output=str(ROOT), source_ownership_and_concurrent_writers='New isolated branch/run in current Linux workspace; prior macOS SSD instructions apply to the old host. Current user-authorized cloud workspace is used; no old run modified.',
        fidelity_tolerances={'projection_symmetry_idempotence':1e-10,'rank22_weight_relative_squared_error':1e-12,'roundtrip_max_logit_abs':2e-4,'roundtrip_ce_abs':2e-6,'duplicate_teacher_max_logit_abs':1e-6},
        canary_matrix='Independent small-array projection/error algebra; full-rank fit identity; official HF teacher repeated; full model spline22 reconstruction against teacher.',
        cost_estimate='4.4GB public checkpoint; 4 CPU threads; aim 10-25 minutes measurements, hard measurement deadline 30 minutes; no paid compute')
    template['evaluation'] = dict(primary_endpoint='16 fixed validation windows x 128 targets = 2048 targets; all token positions scored',confirmation_frozen_before_scoring=False,
        canonical_precision='FP32',duplicate_policy='Teacher first window repeated; spline22 full-validation roundtrip; recompute derived aggregates from per-window values',
        uncertainty_method='Descriptive per-window CE changes and standard error over 16 windows; no independence or population-level guarantee',
        multiple_comparison_policy='All 9 arms published; K8 gate fixed in advance, no formal superiority significance claim',no_test_tuning=True,
        local_ffn_diagnostic='Teacher-normalized FFN inputs and output captures for first training and first validation windows; relative squared FFN output error over all 22 layers. These are diagnostics, no fitting to validation activations.')
    template['deployment'] = dict(weight_dtype='FP32',kv_or_state_dtype='Not measured; cache disabled',quantization_metadata='None',batching='1 x 128',
        prefill_decode_protocol='Not measured: functional teacher-forced evaluation only',warmup_and_synchronization='No deployment timing claim',
        memory_components='Analytical compact basis storage counted; execution materializes all 22 layers and is not a compact inference runtime',latency_statistics='Unmeasured')
    template['evidence'] = dict(protocol_sha256='Stored in REGISTRATION.json to avoid self-reference',source_manifest='SOURCES.json',checkpoint_manifest='INPUTS.json',progress_path='progress.jsonl',
        terminal_receipt_path='TERMINAL.json',preservation_policy='Commit source, protocols, small exact results and charts to user fork; external checkpoint/data remain recoverable from pinned revisions and SHA256. Transformed weights deterministically regenerate from input and saved small projectors.',
        cleanup_receipt='TERMINAL.json; no cloud resources provisioned')
    template['decision'] = dict(outcome='Pending measurement',reason='Pending measurement',next_experiment='Determined from this screen; no follow-up launch implied')
    write('PROTOCOL.json', template)
    source_hashes = {p.name:sha(p) for p in ROOT.glob('*.py')}
    write('SOURCES.json', {'files':source_hashes})
    write('REGISTRATION.json', {'status':'FROZEN_BEFORE_MEASUREMENT','protocol_sha256':sha(ROOT/'PROTOCOL.json'),
          'windows_npz_sha256':sha(args.cache/'windows.npz'),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    registry_path = REPO / 'auto-research/RUN_REGISTRY.json'
    registry = json.loads(registry_path.read_text())
    assert not any(r['run_id']==ROOT.name for r in registry['runs'])
    registry['runs'].append({'run_id':ROOT.name,'status':'FROZEN_BEFORE_MEASUREMENT','protocol':str((ROOT/'PROTOCOL.json').relative_to(REPO)),
        'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'metrics':str((ROOT/'FUNCTIONAL.json').relative_to(REPO)),
        'report':f'auto-research/reports/{ROOT.name}.md','publication_target':'JRowe47/Super-1 main only'})
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    print('FROZEN',sha(ROOT/'PROTOCOL.json'),flush=True)


def check_sources():
    for name, digest in json.loads((ROOT/'SOURCES.json').read_text())['files'].items():
        assert sha(ROOT/name)==digest, name
    reg=json.loads((ROOT/'REGISTRATION.json').read_text())
    assert sha(ROOT/'PROTOCOL.json')==reg['protocol_sha256']


def structure(args):
    import torch
    from safetensors import safe_open
    check_sources()
    start, cpu = time.monotonic(),time.process_time()
    torch.set_num_threads(4)
    all_results={}
    projectors={}
    with safe_open(args.inputs/'model/model.safetensors',framework='pt',device='cpu') as f:
        for family in FAMILIES:
            tensors=[f.get_tensor(f'model.layers.{i}.mlp.{family}.weight').reshape(-1) for i in range(22)]
            gram=np.zeros((22,22),dtype=np.float64)
            width=tensors[0].numel()
            for offset in range(0,width,262144):
                x=np.stack([t[offset:offset+262144].numpy() for t in tensors]).astype(np.float64)
                gram+=x@x.T
            trace=np.trace(gram)
            norms=np.sqrt(np.diag(gram))
            cosine=gram/np.outer(norms,norms)
            centered=(np.eye(22)-np.ones((22,22))/22)@gram@(np.eye(22)-np.ones((22,22))/22)
            eigenvalues=np.linalg.eigvalsh(centered)[::-1]
            rows=[]
            rng=np.random.default_rng(SEED)
            permutations=[rng.permutation(22) for _ in range(64)]
            for k in RANKS:
                for kind in ['pca','spline','cosine']:
                    q,p=depth_basis(kind,k,gram)
                    assert np.max(np.abs(p@p-p))<1e-10
                    error=max(0.,float(1-np.trace(p@gram)/trace))
                    if k==22: assert error<1e-12
                    row={'kind':kind,'rank':k,'relative_squared_weight_error':error,'retained_weight_energy':1-error}
                    if kind in ['spline','cosine']:
                        null=np.asarray([1-np.trace(p[np.ix_(perm,perm)]@gram)/trace for perm in permutations])
                        row.update(shuffled_error_mean=float(null.mean()),shuffled_error_min=float(null.min()),
                                   shuffled_error_max=float(null.max()),shuffled_error_std=float(null.std()),
                                   natural_order_advantage_vs_shuffle=float(null.mean()-error))
                    rows.append(row)
                    projectors[f'{family}_{kind}{k}']=p.tolist()
            all_results[family]={'shape':list(f.get_tensor(f'model.layers.0.mlp.{family}.weight').shape),
                'scalars_per_matrix':width,'gram':gram.tolist(),'cosine_similarity':cosine.tolist(),
                'adjacent_cosine_mean':float(np.diag(cosine,1).mean()),'mean_mode_energy_fraction':float(gram.sum()/22/trace),
                'centered_eigenvalues':eigenvalues.tolist(),'rows':rows}
            event('structure_family',family=family,adjacent_cosine=all_results[family]['adjacent_cosine_mean'],
                  pca8_error=next(r['relative_squared_weight_error'] for r in rows if r['kind']=='pca' and r['rank']==8),
                  spline8_error=next(r['relative_squared_weight_error'] for r in rows if r['kind']=='spline' and r['rank']==8))
            write('STRUCTURE.json',all_results)
    # An independent explicit small-array residual check verifies trace/error algebra.
    rng=np.random.default_rng(SEED)
    x=rng.standard_normal((22,97))
    g=x@x.T
    checks=[]
    for kind in ['pca','spline','cosine']:
        for k in [4,8,20,22]:
            _,p=depth_basis(kind,k,g)
            direct=float(np.linalg.norm(x-p@x)**2/np.linalg.norm(x)**2)
            analytic=float(1-np.trace(p@g)/np.trace(g))
            assert abs(direct-analytic)<1e-12
            checks.append({'kind':kind,'rank':k,'absolute_error':abs(direct-analytic)})
    write('PROJECTORS.json',projectors)
    write('STRUCTURE_CHECKS.json',{'passed':True,'independent_residual_checks':checks,'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu})
    event('structure_complete',wall_seconds=time.monotonic()-start)


def functional(args):
    import torch
    import torch.nn.functional as F
    import psutil
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    check_sources()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    process=psutil.Process()
    allowed=process.cpu_affinity()
    if len(allowed)>=4: process.cpu_affinity(allowed[:4])
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    projectors=json.loads((ROOT/'PROJECTORS.json').read_text())
    structure_result=json.loads((ROOT/'STRUCTURE.json').read_text())
    windows=np.load(args.cache/'windows.npz')
    assert sha(args.cache/'windows.npz')==json.loads((ROOT/'REGISTRATION.json').read_text())['windows_npz_sha256']
    started,cpu_started=time.monotonic(),time.process_time()
    event('load_teacher')
    model=AutoModelForCausalLM.from_pretrained(args.inputs/'model',local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa',use_safetensors=True).eval()
    for p in model.parameters(): p.requires_grad_(False)
    param_count=sum(p.numel() for p in model.parameters())
    per_basis=sum(structure_result[f]['scalars_per_matrix'] for f in FAMILIES)
    ffn_count=22*per_basis
    event('teacher_loaded',parameters=param_count,rss_bytes=process.memory_info().rss)
    captured={}
    for split in ['train','validation']:
        records={}
        hooks=[]
        for index,layer in enumerate(model.model.layers):
            def hook(module,inputs,output,index=index):
                records[index]={'x':inputs[0].detach().clone(),'y':output.detach().clone()}
            hooks.append(layer.mlp.register_forward_hook(hook))
        with torch.inference_mode():
            model(torch.as_tensor(windows[split][0,:-1])[None,:],use_cache=False)
        for hook in hooks: hook.remove()
        captured[split]=records
    teacher_logp=[]
    teacher_logits_first=None
    rows=[]
    failures=[]

    def reconstruct(kind,k,source):
        reconstruction_start=time.monotonic()
        for family in FAMILIES:
            p=torch.tensor(projectors[f'{family}_{kind}{k}'],dtype=torch.float32)
            tensors=[source.get_tensor(f'model.layers.{i}.mlp.{family}.weight').reshape(-1) for i in range(22)]
            n=tensors[0].numel()
            fitted=torch.empty((22,n),dtype=torch.float32)
            with torch.inference_mode():
                for offset in range(0,n,262144):
                    x=torch.stack([t[offset:offset+262144] for t in tensors])
                    torch.mm(p,x,out=fitted[:,offset:offset+x.shape[1]])
            shape=structure_result[family]['shape']
            for i,layer in enumerate(model.model.layers):
                getattr(layer.mlp,family).weight=torch.nn.Parameter(fitted[i].reshape(shape),requires_grad=False)
            del tensors,fitted
        return time.monotonic()-reconstruction_start

    with safe_open(args.inputs/'model/model.safetensors',framework='pt',device='cpu') as source:
        for arm in protocol['design']['arms']:
            if time.monotonic()-started>1800:
                failures.append({'id':arm['id'],'reason':'Measurement deadline reached before arm'})
                continue
            arm_start,arm_cpu=time.monotonic(),time.process_time()
            event('arm_start',arm=arm['id'])
            try:
                reconstruction_seconds=0.
                if arm['kind']!='original':
                    reconstruction_seconds=reconstruct(arm['kind'],arm['rank'],source)
                diagnostics={}
                with torch.inference_mode():
                    for split,records in captured.items():
                        errors=[]
                        for i,layer in enumerate(model.model.layers):
                            actual=layer.mlp(records[i]['x'])
                            expected=records[i]['y']
                            errors.append(float((actual-expected).double().square().sum()/expected.double().square().sum()))
                        diagnostics[split]={'per_layer_relative_squared_output_error':errors,'mean':float(np.mean(errors))}
                ces=[];kls=[];agreements=[];roundtrip_maxdiff=0.
                with torch.inference_mode():
                    for i,window in enumerate(windows['validation']):
                        x=torch.as_tensor(window[:-1])[None,:]
                        y=torch.as_tensor(window[1:])
                        logits=model(x,use_cache=False).logits[0]
                        assert torch.isfinite(logits).all()
                        lp=F.log_softmax(logits,dim=-1)
                        ce=float(F.nll_loss(lp,y))
                        if arm['kind']=='original':
                            teacher_logp.append(lp.clone())
                            if i==0:
                                teacher_logits_first=logits.clone()
                                repeated=model(x,use_cache=False).logits[0]
                                drift=float((repeated-logits).abs().max())
                                assert drift<=protocol['execution']['fidelity_tolerances']['duplicate_teacher_max_logit_abs']
                        else:
                            if arm['id']=='spline22_roundtrip' and i==0:
                                roundtrip_maxdiff=float((logits-teacher_logits_first).abs().max())
                        t=teacher_logp[i]
                        kl=float((t.exp()*(t-lp)).sum(-1).mean())
                        agree=float((t.argmax(-1)==lp.argmax(-1)).float().mean())
                        ces.append(ce);kls.append(kl);agreements.append(agree)
                stored=param_count-ffn_count+arm['rank']*per_basis
                if arm['kind']=='pca': stored+=3*22*(arm['rank']-1)
                row={**arm,'validation_ce':float(np.mean(ces)),'perplexity':float(np.exp(np.mean(ces))),
                     'forward_kl':float(np.mean(kls)),'teacher_top1_agreement':float(np.mean(agreements)),
                     'window_ce':ces,'window_kl':kls,'window_agreement':agreements,'local_ffn_diagnostics':diagnostics,
                     'analytical_stored_scalars':stored,'analytical_fp32_weight_bytes':stored*4,
                     'analytical_weight_saving_fraction':1-stored/param_count,
                     'reconstruction_seconds':reconstruction_seconds,'wall_seconds':time.monotonic()-arm_start,
                     'cpu_seconds':time.process_time()-arm_cpu,'rss_bytes':process.memory_info().rss,
                     'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                if arm['kind']=='original': row['duplicate_first_window_max_logit_abs']=drift
                if arm['id']=='spline22_roundtrip':
                    row['roundtrip_first_window_max_logit_abs']=roundtrip_maxdiff
                    row['roundtrip_ce_abs']=abs(row['validation_ce']-rows[0]['validation_ce'])
                    assert roundtrip_maxdiff<=protocol['execution']['fidelity_tolerances']['roundtrip_max_logit_abs']
                    assert row['roundtrip_ce_abs']<=protocol['execution']['fidelity_tolerances']['roundtrip_ce_abs']
                rows.append(row)
                write('FUNCTIONAL.json',{'rows':rows,'failures':failures,'teacher_parameters':param_count,'teacher_ffn_parameters':ffn_count,
                      'targets_per_arm':2048,'source_checkpoint_sha256':next(v['sha256'] for v in json.loads((ROOT/'INPUTS.json').read_text())['files'] if v['path'].endswith('model.safetensors'))})
                event('arm_complete',arm=arm['id'],ce=row['validation_ce'],kl=row['forward_kl'],agreement=row['teacher_top1_agreement'],
                      local_val_error=diagnostics['validation']['mean'],wall_seconds=row['wall_seconds'])
            except Exception as error:
                failures.append({'id':arm['id'],'error_type':type(error).__name__,'message':str(error)})
                write('FAILURES.json',failures)
                event('arm_failed',arm=arm['id'],error_type=type(error).__name__,message=str(error))
                if arm['id'] in ['teacher','spline22_roundtrip']: raise
    write('TERMINAL.json',{'status':'COMPLETED' if len(rows)==len(protocol['design']['arms']) and not failures else 'INCOMPLETE',
          'completed_arms':len(rows),'declared_arms':len(protocol['design']['arms']),'failures':failures,
          'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu_started,
          'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'official_test_accessed':False,
          'training_updates':0,'cloud_resources_provisioned':False,'followup_running':False})
    event('functional_complete',completed_arms=len(rows),wall_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','structure','functional'])
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.stage](args)
