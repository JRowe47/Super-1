"""Anchored matrix-valued depth splines: weight, activation, and output fitting.

All 22 TinyLlama blocks still execute. Block 0's FFN is exact. The remaining
21 FFNs share K spline controls per tensor family. Low-rank *control-bank*
updates are merged into those same controls; there are no learned per-layer
adapters or additional stored matrices in the counted representation.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PARENT = ROOT.parent / 'tinyllama-depth-curves-20261017-v1'
ANCHOR = ROOT.parent / 'tinyllama-depth-anchor-20261017-v1'
FAILED = ROOT.parent / 'tinyllama-depth-calibration-20261017-v1'
FAMILIES = ['gate_proj', 'up_proj', 'down_proj']
COUNTS = [19, 15]
SEED = 101702
EPOCHS = 4
RANK = 8
BATCH = 64
LR = 0.003
SCALE = 2.0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def event(event_kind, **kw):
    row = {'event': event_kind, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw}
    with (ROOT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row), flush=True)


def basis(k, layers=21):
    knots = np.r_[np.zeros(4), np.arange(1, k-3)/(k-3), np.ones(4)]
    s = BSpline.design_matrix(np.linspace(0, 1, layers), knots, 3).toarray()
    q, _ = np.linalg.qr(s, mode='reduced')
    assert q.shape == (layers, k)
    return q


def mixing(q, d):
    """d[L,C] positive diagonal input second moments -> P[C,L,L]."""
    rhs = q.T[None, :, :] * d.T[:, None, :]
    controls = np.linalg.solve(rhs @ q, rhs)
    return q[None, :, :] @ controls


def delta(x, a, b, q):
    import torch
    k, rank, inputs = b.shape
    z = x @ b.reshape(k*rank, inputs).T
    z = z.reshape(*x.shape[:-1], k, rank) * q[:, None]
    return SCALE * (z.flatten(-2) @ a.permute(0, 2, 1).reshape(k*rank, -1))


def canary():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    q = basis(4, 7)
    d = np.exp(rng.normal(size=(7, 3)))
    p = mixing(q, d)
    x = rng.normal(size=(7, 5, 3))
    fitted = np.einsum('cij,joc->ioc', p, x)
    direct = np.empty_like(fitted)
    for c in range(3):
        root = np.sqrt(d[:, c])
        direct[:, :, c] = q @ np.linalg.lstsq(root[:, None]*q, root[:, None]*x[:, :, c], rcond=None)[0]
    weighted_error = float(np.max(np.abs(fitted-direct)))
    assert weighted_error < 1e-10
    identity_error = float(np.max(np.abs(mixing(basis(7, 7), d)-np.eye(7))))
    assert identity_error < 1e-10
    torch.manual_seed(SEED)
    a = torch.randn(4, 5, 2, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 2, 3, dtype=torch.float64, requires_grad=True)
    xx = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    aa = a.detach().clone().requires_grad_()
    bb = b.detach().clone().requires_grad_()
    xxx = xx.detach().clone().requires_grad_()
    qt = torch.tensor(q[2])
    y = delta(xx, a, b, qt)
    yy = F.linear(xxx, SCALE*torch.einsum('k,kor,kri->oi', qt, aa, bb))
    y.square().mean().backward(); yy.square().mean().backward()
    forward_error = float((y-yy).abs().max().detach())
    grad_error = max(float((u.grad-v.grad).abs().max()) for u, v in [(a, aa), (b, bb), (xx, xxx)])
    assert forward_error < 1e-10 and grad_error < 1e-10
    o = torch.optim.AdamW([a, b], lr=LR, weight_decay=0)
    oo = torch.optim.AdamW([aa, bb], lr=LR, weight_decay=0)
    o.step(); oo.step()
    step_error = max(float((u-v).abs().max().detach()) for u, v in [(a, aa), (b, bb)])
    assert step_error < 1e-10
    save('CANARY.json', {'passed': True, 'weighted_lstsq_max_abs': weighted_error,
        'weighted_full_rank_identity_max_abs': identity_error, 'adapter_forward_max_abs': forward_error,
        'adapter_gradient_max_abs': grad_error, 'adam_step_max_abs': step_error,
        'source_sha256': sha(__file__), 'data': 'Seeded synthetic arrays only; no model measurement'})


def prepare(args):
    import torch
    import transformers
    import scipy
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from transformers.models.llama import modeling_llama
    assert not (ROOT/'PROTOCOL.json').exists(), 'Use a new run directory'
    assert json.loads((ROOT/'CANARY.json').read_text())['source_sha256'] == sha(__file__)
    parent = json.loads((PARENT/'PROTOCOL.json').read_text())
    inputs = json.loads((PARENT/'INPUTS.json').read_text())
    for item in inputs['files']:
        path = args.inputs / ('model' if '/model/' in item['path'] else '') / Path(item['path']).name
        assert sha(path) == item['sha256'], path
    tokenizer = AutoTokenizer.from_pretrained(args.inputs/'model', local_files_only=True)
    old = np.load(args.parent_cache/'windows.npz')
    assert sha(args.parent_cache/'windows.npz') == json.loads((PARENT/'REGISTRATION.json').read_text())['windows_npz_sha256']
    windows, boundaries = {}, {}
    for split in ['train', 'validation']:
        lines = pq.read_table(args.inputs/f'{split}.parquet', columns=['text'])['text'].to_pylist()
        ids = np.asarray(tokenizer('\n\n'.join(t for t in lines if t.strip()), add_special_tokens=False, truncation=False)['input_ids'], dtype=np.int64)
        if split == 'train':
            starts = np.linspace(0, len(ids)-129, 8, dtype=np.int64)
            key = 'calibration'
        else:
            old_starts = parent['data']['split_boundaries']['validation']['starts']
            rng = np.random.default_rng(SEED)
            chosen = []
            while len(chosen) < 16:
                s = int(rng.integers(0, len(ids)-128))
                if all(abs(s-t) >= 129 for t in list(old_starts)+chosen):
                    chosen.append(s)
            starts = np.asarray(sorted(chosen), dtype=np.int64)
            key = 'fresh_validation'
        windows[key] = np.stack([ids[s:s+129] for s in starts])
        boundaries[key] = {'starts': starts.tolist(), 'windows': len(starts), 'targets_per_window': 128,
            'ids_sha256': hashlib.sha256(windows[key].tobytes()).hexdigest(),
            'full_split_token_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest()}
    windows['reused_validation'] = old['validation']
    args.cache.mkdir(parents=True, exist_ok=True)
    assert not (args.cache/'windows.npz').exists()
    np.savez(args.cache/'windows.npz', **windows)
    boundaries['reused_validation'] = parent['data']['split_boundaries']['validation']
    p = json.loads((REPO/'auto-research/PROTOCOL_TEMPLATE.json').read_text())
    p.update(status='FROZEN_BEFORE_MEASUREMENT', experiment_id=ROOT.name, run_id=ROOT.name,
        parent_run_ids=[PARENT.name, ANCHOR.name, FAILED.name],
        hypothesis='At fixed spline-control storage, fitting teacher FFN outputs recovers end-to-end behavior beyond weight-space or diagonal-input fitting, while retaining an exact first FFN.',
        primary_comparator='Anchored K19 raw and diagonal-activation fits; output-calibrated K19 must improve on both. K15 is a declared stronger-compression secondary arm.',
        primary_metric='Mean next-token CE on 16 fresh frozen validation windows (2048 targets), nats/token, 128-token reset context.',
        minimum_useful_effect='Output-calibrated K19 CE improves by >=0.10 nats/token over BOTH K19 raw and activation fits, and finishes within 0.30 of the teacher.',
        success_rule='All primary thresholds, complete arms, and fidelity checks pass. Exploratory continuation only, not confirmation.',
        futility_rule='Abort on nonfinite loss, source/identity/merge fidelity failure, memory failure, or 1800 seconds measurement time. Preserve completed and failed outcomes; no validation-selected checkpoints or hyperparameter retries.')
    p['authorization'] = dict(approved=True, user_approval_reference='User: "Hey, awesome, go ahead with the next level" after prior results and the proposed exact-first-FFN activation/output fitting; earlier explicit PRs to user fork only.',
        approved_resources='Bounded local CPU calibration of low-rank control-bank updates; existing public inputs; no paid service, GPU, large-model pretraining, or official test.',
        max_cost='No paid services', max_walltime='30 minutes combined measurement including the failed v1; excluding source/setup/publication', local_cpu_budget='4 threads, <=120 aggregate CPU minutes')
    arms = [{'id': 'teacher', 'kind': 'original', 'controls': 21}]
    for k in COUNTS:
        arms += [{'id': f'k{k}_{kind}', 'kind': kind, 'controls': k} for kind in ['raw', 'activation', 'output']]
    p['design'] = dict(original_source_hashes={'experiment.py':sha(__file__), 'modeling_llama.py':sha(modeling_llama.__file__),
            'parent_inputs':sha(PARENT/'INPUTS.json'), 'parent_functional':sha(PARENT/'FUNCTIONAL.json'), 'parent_anchor_results':sha(ANCHOR/'RESULTS.json'), 'failed_source':sha(FAILED/'experiment.py'), 'failed_terminal':sha(FAILED/'TERMINAL.json')},
        mechanism_equations='For blocks1..21 and each FFN family, Q is the orthonormalized clamped uniform cubic spline basis. Raw Xhat=QQ^T X. Activation: each input column c uses D_c=diag(E[x_lc^2]) floored at 1e-4 times that layer input mean; Xhat_c=Q(Q^T D_c Q)^-1 Q^T D_c X_c. Down uses ORIGINAL teacher gate/up intermediate second moments. Output: start from activation fit, add sum_k Q_lk*(2*A_fk*B_fk), rank8 per control; minimize normalized teacher FFN output MSE on teacher FFN inputs. Fold updates into the same K matrix controls after training.',
        arms=arms, control_rationale='One exact original FFN plus K full matrices per family, fixed spline geometry. All fitting variants at a K have identical final dense-control scalar counts. Raw and activation have zero optimizer updates; output calibration costs extra training. K19 totals20 FFN equivalents; K15 totals16.',
        seeds=[SEED], seed_selection_method='One predeclared calibration/adapter seed; exploratory, no replication or seed selection.',
        learning_rate_trials_per_arm=1, parameter_counts='Counted from checkpoint and learned adapters; only K19/K15 output arms train. Adapters fold into existing banks and do not increase final control storage.',
        matching_track='Final analytical dense-control parameter matching, not equal fitting compute, deployed memory, or latency.',
        compute_accounting='Wall/process CPU per stage/arm and epoch. ru_maxrss retained as a runtime receipt, not compact-model memory.')
    p['data'] = dict(dataset_versions=parent['data']['dataset_versions'], tokenizer_sha256=parent['data']['tokenizer_sha256'],
        train_sha256=parent['data']['train_sha256'], selection_sha256=parent['data']['selection_sha256'], confirmation_sha256='None; no independent confirmation dataset',
        split_boundaries=boundaries, research_wide_region_reuse_audit='Prior outcomes observed before this design. Fresh validation positions are disjoint from prior 16 windows and frozen before this run, but drawn from the same previously used official validation split; not globally untouched confirmation. Reused windows reported separately.',
        teacher_and_objective='Public pinned TinyLlama teacher only; no OpenAI outputs. Teacher-normalized FFN inputs and FFN outputs from 1024 training token positions. No validation activations used for fitting.', official_test_access_permitted=False)
    p['training'] = dict(updates=EPOCHS*16*21, target_exposures_per_trial=EPOCHS*1024*21,
        tokens_per_update=64, microbatch='64 cached teacher FFN token inputs for one non-anchor block', gradient_accumulation=1,
        bptt_length='Local FFN only; no backpropagation through attention or other blocks', accessible_history='Teacher inputs generated at reset context <=128',
        state_carry_and_detach='No cross-window state; cached detached teacher activations', document_reset_policy='Nonempty lines joined by two newlines, fixed 129-token windows may cross article boundaries.',
        optimizer_schedule={'name':'AdamW','lr':LR,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0,'clip_grad_norm':1.0,
            'schedule':'Cosine from LR to 0.1*LR across all updates','epochs':EPOCHS,'rank':RANK,'adapter_scale':SCALE,
            'initialization':'A=zeros, B=Normal(0,1/sqrt(input_width)), same seed reset for each K',
            'order':'Seeded token permutation each epoch; seeded layer permutation each 64-token minibatch; each layer sees every calibration position once per epoch.'},
        precision='FP64 basis and input moments; FP32 model, adapters, optimizer, merge and evaluation', checkpoint_selection_rule='Final epoch only; no early stopping or best-validation selection; no LR tuning')
    p['execution'] = dict(hardware={'platform':platform.platform(),'logical_cpus':os.cpu_count(),'threads':4,'gpu_used':False},
        runtime_versions={'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'scipy':scipy.__version__},
        resolved_writable_output=str(ROOT), source_ownership_and_concurrent_writers='New isolated run on existing user-fork branch in explicitly authorized Linux workspace; historical macOS mount restrictions do not describe this host.',
        fidelity_tolerances={'synthetic_math_abs':1e-10,'teacher_reused_ce_abs':2e-6,'anchor_sha256':'exact','merge_relative_squared_output_error':1e-9,'merged_curve_relative_squared_residual':1e-10},
        canary_matrix='Independent weighted least squares, full-rank identity, factored-vs-dense adapter forward/input/parameter gradients and complete AdamW step; full-size anchor hashes, teacher replay, all-layer merge and depth-subspace checks.',
        cost_estimate='Existing checkpoint/data; 4 CPU threads; <=30 minutes; rank8 updates only, no GPU.')
    p['evaluation'] = dict(primary_endpoint='2048 fresh validation targets; reused 2048 validation targets secondary; full-vocabulary teacher KL and top1 agreement.', confirmation_frozen_before_scoring=False,
        canonical_precision='FP32',duplicate_policy='Teacher reused-window CE must reproduce parent; repeat first fresh teacher window; compare premerge vs merged FFN outputs on 64 calibration inputs for every non-anchor layer.',
        uncertainty_method='Paired per-window CE differences and descriptive standard errors; one training seed, 16 windows per region; no population or confirmation claim.',
        multiple_comparison_policy='All seven arms retained; K19 primary and K15 secondary declared before measurement; no adaptive retuning.',no_test_tuning=True)
    p['deployment'] = dict(weight_dtype='FP32',kv_or_state_dtype='Cache disabled',quantization_metadata='None',batching='1 x128 for end-to-end evaluation',
        prefill_decode_protocol='Not a deployment benchmark',warmup_and_synchronization='No inference-speed claim',
        memory_components='Analytical control storage only. Model weights are materialized for evaluation. Extra train-time rank8 factors fold into existing controls.',latency_statistics='Unmeasured')
    p['evidence'] = dict(protocol_sha256='REGISTRATION.json',source_manifest='PROTOCOL.json design.original_source_hashes',checkpoint_manifest='CHECKPOINTS.json',
        progress_path='progress.jsonl',terminal_receipt_path='TERMINAL.json',
        preservation_policy='Keep immutable source/protocol/metrics and exact FP32 adapter factors; deterministic original-weight projection plus saved factors reconstructs final weights. Public checkpoint/corpus remain at pinned sources with hashes; do not redistribute corpus text.',cleanup_receipt='TERMINAL.json')
    p['decision'] = dict(outcome='Pending',reason='Pending measurements',next_experiment='Decide from results; no automatic follow-up launch')
    p['recovery'] = {'predecessor':FAILED.name,'failure':'Logging event() kind argument collided with kind keyword after first raw projection; only teacher was scored; no optimizer updates occurred.', 'changes':'Rename logger positional argument only; preserve fitting/data/seed/optimizer/thresholds. Deduct failed-run time from shared budget.', 'predecessor_wall_seconds':json.loads((FAILED/'TERMINAL.json').read_text())['wall_seconds']}
    save('PROTOCOL.json', p)
    save('REGISTRATION.json', {'status':'FROZEN_BEFORE_MEASUREMENT','protocol_sha256':sha(ROOT/'PROTOCOL.json'),
        'windows_npz_sha256':sha(args.cache/'windows.npz'),'canary_sha256':sha(ROOT/'CANARY.json'),
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    path = REPO/'auto-research/RUN_REGISTRY.json'
    registry = json.loads(path.read_text())
    assert not any(r['run_id']==ROOT.name for r in registry['runs'])
    registry['runs'].append({'run_id':ROOT.name,'status':'FROZEN_BEFORE_MEASUREMENT','protocol':str((ROOT/'PROTOCOL.json').relative_to(REPO)),
        'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'metrics':str((ROOT/'RESULTS.json').relative_to(REPO)),
        'report':f'auto-research/reports/{ROOT.name}.md','publication_target':'JRowe47/Super-1 main only',
        'approval':p['authorization']['user_approval_reference']})
    path.write_text(json.dumps(registry,indent=2)+'\n')
    print('FROZEN',sha(ROOT/'PROTOCOL.json'),flush=True)


def run(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    from safetensors.torch import save_file
    p = json.loads((ROOT/'PROTOCOL.json').read_text())
    reg = json.loads((ROOT/'REGISTRATION.json').read_text())
    assert sha(__file__) == p['design']['original_source_hashes']['experiment.py']
    assert sha(ROOT/'PROTOCOL.json') == reg['protocol_sha256']
    assert sha(args.cache/'windows.npz') == reg['windows_npz_sha256']
    assert not (ROOT/'RESULTS.json').exists(), 'Use a new run; completed measurements are immutable'
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('highest')
    start, cpu = time.monotonic(), time.process_time()
    def deadline():
        if time.monotonic()-start + p['recovery']['predecessor_wall_seconds'] > 1800:
            raise TimeoutError('Frozen 30-minute measurement budget')
    model = AutoModelForCausalLM.from_pretrained(args.inputs/'model',local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa',use_safetensors=True).eval()
    for v in model.parameters(): v.requires_grad_(False)
    windows = dict(np.load(args.cache/'windows.npz'))
    nparams = sum(v.numel() for v in model.parameters())
    one_ffn = sum(getattr(model.model.layers[0].mlp,f).weight.numel() for f in FAMILIES)
    def anchor_hashes():
        return {f:hashlib.sha256(getattr(model.model.layers[0].mlp,f).weight.detach().numpy().tobytes()).hexdigest() for f in FAMILIES}
    original_anchor = anchor_hashes()
    rows, checks, checkpoint_records = [], [], []
    calibration_x = [[] for _ in range(21)]
    calibration_y = [[] for _ in range(21)]
    moments_x = torch.zeros(21,2048,dtype=torch.float64)
    moments_h = torch.zeros(21,5632,dtype=torch.float64)
    handles = []
    for l, block in enumerate(model.model.layers[1:]):
        def hook(module, inputs, output, i=l):
            x=inputs[0].detach().reshape(-1,2048)
            y=output.detach().reshape(-1,2048)
            calibration_x[i].append(x.clone()); calibration_y[i].append(y.clone())
            moments_x[i].add_(x.double().square().sum(0))
        def down_hook(module, inputs, i=l):
            moments_h[i].add_(inputs[0].detach().reshape(-1,5632).double().square().sum(0))
        handles += [block.mlp.register_forward_hook(hook),block.mlp.down_proj.register_forward_pre_hook(down_hook)]
    with torch.no_grad():
        for w in windows['calibration']:
            model(torch.as_tensor(w[:-1])[None,:],use_cache=False)
    for h in handles: h.remove()
    calibration_x = [torch.cat(x) for x in calibration_x]
    calibration_y = [torch.cat(y) for y in calibration_y]
    normalization = [y.square().mean() for y in calibration_y]
    moments = {'gate_proj':moments_x.numpy()/1024,'up_proj':moments_x.numpy()/1024,'down_proj':moments_h.numpy()/1024}
    np.savez_compressed(ROOT/'INPUT_MOMENTS.npz', **moments)
    save('CALIBRATION.json',{'token_positions':1024,'layers':list(range(1,22)),
        'input_moments_sha256':sha(ROOT/'INPUT_MOMENTS.npz'),'output_mean_square':[float(v) for v in normalization],
        'collection_wall_seconds':time.monotonic()-start})
    event('calibration_collected',tokens=1024)

    def evaluate(arm):
        deadline()
        t0,c0=time.monotonic(),time.process_time()
        metrics={}
        for region in ['fresh_validation','reused_validation']:
            ce,kl,agreement=[],[],[]
            with torch.no_grad():
                for index,w in enumerate(windows[region]):
                    logits=model(torch.as_tensor(w[:-1])[None,:],use_cache=False).logits[0]
                    assert torch.isfinite(logits).all()
                    lp=F.log_softmax(logits,dim=-1)
                    target=args.cache/f'teacher_{region}_{index}.npy'
                    if arm['id']=='teacher':
                        np.save(target,lp.numpy())
                        if region=='fresh_validation' and index==0:
                            again=model(torch.as_tensor(w[:-1])[None,:],use_cache=False).logits[0]
                            diff=float((logits-again).abs().max())
                            assert diff<=1e-6
                            checks.append({'kind':'teacher_duplicate','max_logit_abs':diff})
                    teacher=torch.from_numpy(np.load(target))
                    ce.append(float(F.nll_loss(lp,torch.as_tensor(w[1:]))))
                    kl.append(float((teacher.exp()*(teacher-lp)).sum(-1).mean()))
                    agreement.append(float((teacher.argmax(-1)==lp.argmax(-1)).float().mean()))
            metrics[region]={'ce':float(np.mean(ce)),'perplexity':float(np.exp(np.mean(ce))),
                'forward_kl':float(np.mean(kl)),'teacher_top1_agreement':float(np.mean(agreement)),
                'window_ce':ce,'window_kl':kl,'window_agreement':agreement,'targets':2048}
        if arm['id']=='teacher':
            old=next(r for r in json.loads((PARENT/'FUNCTIONAL.json').read_text())['rows'] if r['id']=='teacher')
            diff=float(np.max(np.abs(np.array(metrics['reused_validation']['window_ce'])-old['window_ce'])))
            assert diff<=2e-6
            checks.append({'kind':'teacher_parent_replay','max_window_ce_abs':diff})
        assert anchor_hashes()==original_anchor
        stored=nparams if arm['id']=='teacher' else nparams-(21-arm['controls'])*one_ffn
        row={**arm,'metrics':metrics,'analytical_stored_scalars':stored,'analytical_weight_saving_fraction':1-stored/nparams,
            'evaluation_wall_seconds':time.monotonic()-t0,'evaluation_cpu_seconds':time.process_time()-c0,
            'anchor_byte_exact':True}
        rows.append(row)
        save('RESULTS.json',{'rows':rows,'teacher_parameters':nparams,'one_ffn_parameters':one_ffn})
        save('CHECKS.json',{'checks':checks})
        event('evaluated',arm=arm['id'],fresh_ce=metrics['fresh_validation']['ce'],reused_ce=metrics['reused_validation']['ce'])

    def project(k, kind):
        deadline()
        q=basis(k)
        t0,c0=time.monotonic(),time.process_time()
        with safe_open(args.inputs/'model/model.safetensors',framework='pt',device='cpu') as source:
            for family in FAMILIES:
                src=[source.get_tensor(f'model.layers.{l}.mlp.{family}.weight') for l in range(1,22)]
                shape=src[0].shape
                if kind=='raw':
                    mix=torch.tensor(q@q.T,dtype=torch.float32)
                    fitted=torch.empty(21,src[0].numel(),dtype=torch.float32)
                    with torch.no_grad():
                        for offset in range(0,src[0].numel(),262144):
                            x=torch.stack([s.reshape(-1)[offset:offset+262144] for s in src])
                            torch.mm(mix,x,out=fitted[:,offset:offset+x.shape[1]])
                    fitted=fitted.reshape(21,*shape)
                else:
                    d=moments[family]
                    d=np.maximum(d,1e-4*d.mean(axis=1,keepdims=True))
                    mix=torch.tensor(mixing(q,d),dtype=torch.float32)
                    fitted=torch.empty(21,*shape,dtype=torch.float32)
                    with torch.no_grad():
                        for offset in range(0,shape[1],64):
                            x=torch.stack([s[:,offset:offset+64] for s in src])
                            part=torch.bmm(mix[offset:offset+x.shape[2]],x.permute(2,0,1))
                            fitted[:,:,offset:offset+x.shape[2]]=part.permute(1,2,0)
                for l,block in enumerate(model.model.layers[1:]):
                    getattr(block.mlp,family).weight=torch.nn.Parameter(fitted[l],requires_grad=False)
                del fitted,src,mix,x
                gc.collect()
        assert anchor_hashes()==original_anchor
        event('projected',controls=k,kind=kind,wall_seconds=time.monotonic()-t0,cpu_seconds=time.process_time()-c0)

    def local_errors(bank=None,q=None):
        values=[]
        with torch.no_grad():
            for l,block in enumerate(model.model.layers[1:]):
                total=0.
                for off in range(0,1024,128):
                    x=calibration_x[l][off:off+128]
                    y=block.mlp(x) if bank is None else adapted(block.mlp,x,bank,q[l])
                    total+=float((y-calibration_y[l][off:off+128]).square().sum())
                values.append(total/float(calibration_y[l].square().sum()))
        return values

    def adapted(mlp,x,bank,qrow):
        gate=mlp.gate_proj(x)+delta(x,bank['gate_proj_A'],bank['gate_proj_B'],qrow)
        up=mlp.up_proj(x)+delta(x,bank['up_proj_A'],bank['up_proj_B'],qrow)
        h=F.silu(gate)*up
        return mlp.down_proj(h)+delta(h,bank['down_proj_A'],bank['down_proj_B'],qrow)

    def train_and_merge(k):
        deadline()
        t0,c0=time.monotonic(),time.process_time()
        torch.manual_seed(SEED)
        rng=np.random.default_rng(SEED)
        q=torch.tensor(basis(k),dtype=torch.float32)
        bank=torch.nn.ParameterDict()
        for f in FAMILIES:
            out,inp=getattr(model.model.layers[1].mlp,f).weight.shape
            bank[f'{f}_A']=torch.nn.Parameter(torch.zeros(k,out,RANK))
            bank[f'{f}_B']=torch.nn.Parameter(torch.randn(k,RANK,inp)/math.sqrt(inp))
        before=local_errors()
        optimizer=torch.optim.AdamW(bank.parameters(),lr=LR,betas=(0.9,0.999),eps=1e-8,weight_decay=0)
        steps=0;history=[]
        total_steps=EPOCHS*16*21
        for epoch in range(EPOCHS):
            order=rng.permutation(1024)
            losses=[]
            for off in range(0,1024,BATCH):
                idx=torch.tensor(order[off:off+BATCH])
                for l in rng.permutation(21):
                    deadline()
                    rate=LR*(0.1+0.9*0.5*(1+math.cos(math.pi*steps/(total_steps-1))))
                    for group in optimizer.param_groups:group['lr']=rate
                    optimizer.zero_grad(set_to_none=True)
                    predicted=adapted(model.model.layers[l+1].mlp,calibration_x[l][idx],bank,q[l])
                    loss=F.mse_loss(predicted,calibration_y[l][idx])/normalization[l]
                    assert torch.isfinite(loss)
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(bank.parameters(),1.0,error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()));steps+=1
            row={'epoch':epoch+1,'updates':steps,'mean_online_normalized_mse':float(np.mean(losses)),
                'last_lr':rate,'wall_seconds':time.monotonic()-t0}
            history.append(row);event('trained_epoch',controls=k,**row)
        after=local_errors(bank,q)
        checkpoint=ROOT/f'k{k}_adapter.safetensors'
        save_file({key:value.detach().contiguous() for key,value in bank.items()},str(checkpoint))
        checkpoint_records.append({'id':f'k{k}_output','file':checkpoint.name,'sha256':sha(checkpoint),
            'bytes':checkpoint.stat().st_size,'trainable_scalars':sum(v.numel() for v in bank.values()),'updates':steps,
            'reconstruction':'Regenerate activation-weighted base from pinned original and INPUT_MOMENTS.npz; add 2*sum_k Q_lk A_fk B_fk to each non-anchor FFN matrix.'})
        save('CHECKPOINTS.json',{'checkpoints':checkpoint_records})
        premerge=[]
        with torch.no_grad():
            for l,block in enumerate(model.model.layers[1:]):
                premerge.append(adapted(block.mlp,calibration_x[l][:64],bank,q[l]).clone())
            for family in FAMILIES:
                a=bank[f'{family}_A'];b=bank[f'{family}_B']
                bflat=b.reshape(k*RANK,-1)
                for l,block in enumerate(model.model.layers[1:]):
                    aflat=(a*q[l,:,None,None]).permute(1,0,2).reshape(a.shape[1],k*RANK)
                    getattr(block.mlp,family).weight.addmm_(aflat,bflat,alpha=SCALE)
            merge_errors=[]
            for l,block in enumerate(model.model.layers[1:]):
                actual=block.mlp(calibration_x[l][:64])
                error=float((actual-premerge[l]).square().sum()/premerge[l].square().sum())
                assert error<=1e-9
                merge_errors.append(error)
            # Full weight-subspace residual, accumulated in FP64 from FP32 matrices.
            residuals={}
            projector=torch.tensor(basis(k)@basis(k).T,dtype=torch.float64)
            for family in FAMILIES:
                ws=[getattr(block.mlp,family).weight.reshape(-1) for block in model.model.layers[1:]]
                residual,total=0.,0.
                for off in range(0,ws[0].numel(),65536):
                    x=torch.stack([w[off:off+65536] for w in ws]).double()
                    residual+=float((x-projector@x).square().sum());total+=float(x.square().sum())
                residuals[family]=residual/total
                assert residuals[family]<=1e-10
        assert anchor_hashes()==original_anchor
        checks.append({'kind':'output_merge_and_curve','controls':k,'per_layer_merge_relative_squared_error':merge_errors,
            'curve_relative_squared_residual':residuals,'anchor_byte_exact':True})
        save(f'k{k}_TRAINING.json',{'steps':steps,'epochs':history,'before_per_layer_relative_mse':before,
            'after_per_layer_relative_mse':after,'wall_seconds':time.monotonic()-t0,'cpu_seconds':time.process_time()-c0,
            'trainable_scalars':sum(v.numel() for v in bank.values()),'seed':SEED})
        del bank,optimizer,premerge
        gc.collect()

    try:
        evaluate(p['design']['arms'][0])
        for k in COUNTS:
            for kind in ['raw','activation','output']:
                if kind=='output':train_and_merge(k)
                else:project(k,kind)
                local=local_errors()
                save(f'k{k}_{kind}_LOCAL.json',{'per_layer_relative_mse':local,'mean':float(np.mean(local)),
                    'data':'1024 training positions; teacher inputs; diagnostics only for raw/activation'})
                evaluate({'id':f'k{k}_{kind}','kind':kind,'controls':k})
        save('TERMINAL.json',{'status':'COMPLETED','completed_arms':len(rows),'declared_arms':len(p['design']['arms']),
            'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
            'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'optimizer_updates':2*EPOCHS*16*21,'official_test_accessed':False,'cloud_resources_provisioned':False,'followup_running':False})
    except Exception as exc:
        save('TERMINAL.json',{'status':'FAILED','error':repr(exc),'completed_arms':len(rows),
            'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
            'official_test_accessed':False,'cloud_resources_provisioned':False,'followup_running':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['canary','prepare','run'])
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--parent-cache',type=Path)
    args=parser.parse_args()
    if args.stage=='canary':canary()
    elif args.stage=='prepare':prepare(args)
    else:run(args)
