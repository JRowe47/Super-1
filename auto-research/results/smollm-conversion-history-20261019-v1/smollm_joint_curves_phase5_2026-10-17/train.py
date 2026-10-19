"""Joint distillation of windows 4–7 and 12–15 with matched batches."""
import gc,json,math,platform,resource,time
import numpy as np
import torch
from filelock import FileLock
from torch.nn import functional as F
from common import ROOT,OUT,SEED,ARMS,dump,setup,load_model,hashes,data
from curve_model import restore
from joint_model import make_banks,set_model,save_banks,load_banks
from evaluation import evaluate

UPDATES=600


def protocol():
    cal=json.loads((OUT/'calibration.json').read_text())
    spec={'phase':5,'windows_zero_indexed':[[4,5,6,7],[12,13,14,15]],'initialization':'Frozen original knots plus phase-4 selected slopes; all arms have exactly matching initial functions at both reduced resolutions.',
        'arms':ARMS,'primary_curve_arm':'orthogonal_curve','training_resolutions':[3,2],
        'raw_curve':'Train increments of the four Hermite derivatives in original coordinates.',
        'orthogonal_curve':'Same four-derivative function space reparameterized by B=V diag(1/sigma) from the 5x4 derivative sampling matrix A; AB has orthonormal columns.',
        'dense':'Five independent dense sampled weight sets per window: 3-step and 2-step controls have separate Parameters and Adam state; inactive resolution does not update. Fixed residual branch multipliers match curve initialization.',
        'shared_native_blocks':'All 22 blocks outside the two windows, embedding, head and final norm remain frozen; gradients propagate through the complete suffix.',
        'objective':'Mean teacher-to-student next-token KL, temperature 1; no intermediate-state loss in this phase.',
        'updates_per_arm':UPDATES,'updates_per_resolution':UPDATES//2,'batch_sequences':2,'context':128,'seed':SEED,
        'data_exposure':'Exactly the same 600 ordered minibatches and alternating resolutions for all arms; 153600 input tokens per arm, 76800 at each resolution.',
        'optimizer':{'name':'AdamW','weight_decay':0,'betas':[.9,.999],'gradient_clip_norm':1,'schedule':'20-update linear warmup then cosine to 20% of peak'},
        'peak_lrs':{k:v['training_peak_lr'] for k,v in cal['arms'].items()},
        'calibration':'Training-only two-probe RMS update-scale matching to dense LR 1e-4. Probe updates discarded; no validation sweep. This is not exact optimizer equivalence.',
        'checkpoint_selection':'Every 100 updates including initialization: minimum equal mean N=3/N=2 teacher KL on the first 8 validation windows; one shared checkpoint per arm, including dense.',
        'final_validation':'All 32 validation windows; gate delta NLL <= .05 and KL <= .05.',
        'primary_audit':'64 document-held-out windows from 32 previously unused official-train articles. Reserved before training; never used for learning, calibration or selection.',
        'audit_candidates':'All 3 arms at N=3 and N=2; phase4 initial functions at both N; dense weights projected into the Hermite curve space at both N; each individual converted window of orthogonal and dense N=3 as diagnostic ablations.',
        'projection_control':'After checkpoint selection, project dense learned increments D onto the rank-4 curve sample space with U U^T D. No retraining or test-based choices. Measures the immediate cost of imposing this curve constraint on this fitted dense solution, not an optimum over all curves.',
        'long_context_candidates':'Phase4, orthogonal, dense at N=3, context 256, exactly same 128 target tokens as short audit.',
        'parameter_comparison':'Curves: 28320768 trainable increments. Dense: 35400960 across both resolutions (21240576 active N=3; 14160384 active N=2). Dense has 25% more total trained values; update count and token exposure matched, not storage or wall clock.',
        'cost_protocol':'12 alternating native/candidate paired prefill timings after warmup; compare streamed function weights to native blocks with sampled weights and residual scaling folded into o/down matrices.',
        'limits':'One seed; phase4 warm start; short continuation budget; depth-only coefficients. No adaptive depth per sequence, no hierarchical state, no input-conditioned basis. Not a GSA reproduction. Improvement over phase4 cannot isolate joint training from additional updates/data.'}
    dump(OUT/'protocol.json',spec);return spec


@torch.no_grad()
def score(model,original,banks,windows,reference):
    result={}
    for n in [3,2]:
        set_model(model,original,banks,n)
        m=evaluate(model,windows[:8],reference)
        result[str(n)]={k:m[k] for k in ['mean_kl','delta_nll','perplexity_ratio_to_original']}
    restore(model,original);result['mean_kl']=float(np.mean([result[str(n)]['mean_kl'] for n in [3,2]]))
    return result


def train_arm(model,original,kind,windows,trainref,valref,indices,peak_lr):
    # Prevent the optional dense worker and the main loop from updating the
    # same checkpoint concurrently. After waiting, completion is checked again.
    with FileLock(str(ROOT/'cache'/f'train_{kind}.lock')):
        return _train_arm(model,original,kind,windows,trainref,valref,indices,peak_lr)


def _train_arm(model,original,kind,windows,trainref,valref,indices,peak_lr):
    path=OUT/f'training_{kind}.json';checkpoint=ROOT/'checkpoints'/f'{kind}.safetensors'
    if path.exists() and json.loads(path.read_text()).get('status')=='complete':return
    banks=make_banks(original,kind);optimizer=torch.optim.AdamW(banks.parameters(),lr=peak_lr,weight_decay=0)
    best=float('inf');beststep=0;history=[];begin=time.perf_counter();loss_rows=[]
    for step in range(UPDATES+1):
        if step%100==0:
            metrics=score(model,original,banks,windows['validation'],valref)
            history.append({'update':step,'validation':metrics,'elapsed_seconds':time.perf_counter()-begin})
            if metrics['mean_kl']<best:
                best=metrics['mean_kl'];beststep=step;save_banks(banks,checkpoint)
            dump(path,{'status':'training','kind':kind,'history':history,'loss_batches':loss_rows})
            print('VALIDATE',kind,step,metrics,flush=True)
        if step==UPDATES:break
        idx=indices[step];n=[3,2][step%2]
        schedule=(step+1)/20 if step<20 else .2+.8*.5*(1+math.cos(math.pi*(step-20)/(UPDATES-20)))
        for group in optimizer.param_groups:group['lr']=peak_lr*schedule
        optimizer.zero_grad(set_to_none=True);set_model(model,original,banks,n)
        logits=model(windows['train'][idx,:-1],use_cache=False).logits
        with torch.no_grad():lp=F.log_softmax(model.lm_head(trainref['hidden'][idx]),-1)
        loss=(lp.exp()*(lp-F.log_softmax(logits,-1))).sum(-1).mean()
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(banks.parameters(),1.)
        assert torch.isfinite(loss) and torch.isfinite(grad),(kind,step)
        optimizer.step()
        if (step+1)%20==0:
            row={'update':step+1,'steps_per_window':n,'loss':float(loss.detach()),'lr':peak_lr*schedule,'preclip_grad_norm':float(grad)}
            loss_rows.append(row);print('UPDATE',kind,row,flush=True)
    restore(model,original)
    summary={'status':'complete','kind':kind,'history':history,'loss_batches':loss_rows,
        'best_update':beststep,'best_mean_validation_kl':best,'peak_lr':peak_lr,
        'trainable_values':sum(p.numel() for p in banks.parameters()),
        'elapsed_seconds':time.perf_counter()-begin,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    dump(path,summary);print('TRAINED',kind,summary['elapsed_seconds'],flush=True)
    del optimizer,banks;gc.collect()


if __name__=='__main__':
    setup();p=protocol();windows=data();model=load_model();original=list(model.model.layers);before=hashes(model)
    trainref=torch.load(ROOT/'cache/train.pt',weights_only=True);valref=torch.load(ROOT/'cache/validation.pt',weights_only=True)
    indices=torch.randint(0,len(windows['train']),(UPDATES,2),generator=torch.Generator().manual_seed(SEED))
    np.save(ROOT/'data/training_indices.npy',indices.numpy())
    for arm in ARMS:train_arm(model,original,arm,windows,trainref,valref,indices,p['peak_lrs'][arm])
    restore(model,original);assert before==hashes(model)
    dump(OUT/'integrity.json',{'all_272_original_tensor_hashes_unchanged':True,'parameter_sha256':before})
    dump(OUT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'precision':'FP32','device':'CPU','max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
