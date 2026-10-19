"""Train only Hermite derivatives; native weight values remain frozen."""
import argparse, gc, json, resource, time
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from common import ROOT,OUT,WINDOWS,SEED,dump,setup,load_model,hashes,data
from curve_model import HermiteBank,install,restore,window_forward,path_loss
from evaluation import evaluate


def save_bank(bank,path):
    save_file({n:p.detach().contiguous() for n,p in bank.slopes.items()},str(path))


def load_bank(bank,path):
    bank.slopes.load_state_dict(load_file(str(path)))


@torch.no_grad()
def local_score(model,bank,teacher,floor):
    result={}
    for n in [3,2]:
        values=[]
        for j in range(0,8,2):
            batch=teacher[j:j+2]
            predictions=window_forward(model,bank,batch[:,0],n)
            values.append(float(path_loss(predictions,batch,floor)))
        result[str(n)]=float(np.mean(values))
    result['mean']=float(np.mean(list(result.values())))
    return result


@torch.no_grad()
def global_score(model,original,bank,start,windows,reference):
    result={}
    for n in [3,2]:
        install(model,original,{start:{'bank':bank,'steps':n}})
        metrics=evaluate(model,windows[:8],reference)
        result[str(n)]={k:metrics[k] for k in ['mean_kl','delta_nll','perplexity_ratio_to_original']}
    restore(model,original)
    result['mean_kl']=float(np.mean([result[str(n)]['mean_kl'] for n in [3,2]]))
    return result


def protocol():
    p={'name':'Phase 4: joint four-block Hermite curves with exact original knots',
       'windows_zero_indexed':[[4,5,6,7],[12,13,14,15]],'precision':'CPU float32','threads':4,'seed':SEED,
       'curve':'Independent homologous tensor families. Original 4 weight knots fixed; four trainable Hermite slopes per family. Catmull-Rom initial slopes with duplicated boundary neighbors.',
       'coordinates':'t_i=(i+0.5)*4/N-0.5, scale attention and MLP residual branches by 4/N.',
       'resolutions':[4,3,2],'training_resolutions_alternated':[3,2],
       'local_updates':300,'global_updates':60,'batch_sequences':2,'context_tokens':128,
       'local_optimizer':{'name':'AdamW','lr':.001,'weight_decay':0,'clip_norm':1},
       'global_optimizer':{'name':'AdamW','lr':.0001,'weight_decay':0,'clip_norm':1},
       'local_loss':'.5 endpoint plus .5 average path MSE, each divided by per-token teacher displacement energy; floor .001 * median training endpoint displacement energy.',
       'teacher_path':'Linear interpolation between frozen teacher block outputs at fractional intermediate boundaries; a proxy target, not a known continuous teacher trajectory.',
       'global_loss':'Teacher-to-student next-token KL plus 0.1 * local path loss; backpropagate through frozen suffix.',
       'checkpoint_selection':'Local: minimum mean 3/2-step path loss on first 8 validation windows at 0,50,...,300. Global: minimum mean 3/2-step KL on same 8 windows at 0,20,40,60, including local stage checkpoint.',
       'validation':'All 32 windows for final gate: delta NLL <= .05 and teacher KL <= .05.',
       'selection':'Per window, fewest steps passing gate then KL; if neither reduced version passes, lowest-KL reduced version labelled failed-gate representative. Composite of selected representatives locked before test.',
       'test':'First 64 phase-3 test windows, 8192 predicted tokens; test reused across phases, not fresh.',
       'drop_controls':{'3_steps_keep_offsets':[0,1,3],'2_steps_keep_offsets':[0,3]},
       'extra_storage':'14,160,384 trainable FP32 slope values per four-block curve, in addition to the retained original model. No weight compression claimed.',
       'scope_limits':'One seed, short context, fixed resolution per sequence, depth-only coefficients, no neuron alignment, no input conditioning, no hierarchical cache, no GSA reproduction.',
       'test_configurations':'Initial curves at 3/2; trained curves at 4/3/2; prespecified drop controls at 3/2 for both windows; one validation-selected composite.'}
    dump(OUT/'protocol.json',p);return p


def train_one(model,original,start,traincache,valcache,windows):
    name=f'window_{start}';finalpath=ROOT/'checkpoints'/f'{name}.safetensors';reportpath=OUT/f'training_{name}.json'
    if finalpath.exists() and reportpath.exists() and json.loads(reportpath.read_text()).get('status')=='complete':return
    bank=HermiteBank(original[start:start+4]);teacher=traincache[name];valteacher=valcache[name]
    floor=float((teacher[:,-1]-teacher[:,0]).square().mean(-1).median())*.001
    generator=torch.Generator().manual_seed(SEED);indices=torch.randint(0,len(teacher),(360,2),generator=generator)
    history=[];begin=time.perf_counter();best=float('inf');beststep=0
    localpath=ROOT/'checkpoints'/f'{name}_local.safetensors'
    optimizer=torch.optim.AdamW(bank.parameters(),lr=.001,weight_decay=0)
    for step in range(301):
        if step%50==0:
            score=local_score(model,bank,valteacher,floor)
            row={'stage':'local','update':step,'validation':score,'elapsed_seconds':time.perf_counter()-begin}
            history.append(row)
            if score['mean']<best:
                best=score['mean'];beststep=step;save_bank(bank,localpath)
            dump(reportpath,{'status':'training','window':start,'history':history})
            print('LOCAL',start,step,round(score['mean'],6),flush=True)
        if step==300:break
        batch=teacher[indices[step]];n=[3,2][step%2]
        optimizer.zero_grad(set_to_none=True)
        predictions=window_forward(model,bank,batch[:,0],n)
        loss=path_loss(predictions,batch,floor);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(bank.parameters(),1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
        if (step+1)%10==0:print('UPDATE',start,'local',step+1,float(loss.detach()),flush=True)
    del optimizer;load_bank(bank,localpath);gc.collect()
    localbest={'update':beststep,'mean_validation_path_loss':best}
    optimizer=torch.optim.AdamW(bank.parameters(),lr=.0001,weight_decay=0)
    best=float('inf');beststep=0
    for step in range(61):
        if step%20==0:
            score=global_score(model,original,bank,start,windows['validation'],valcache)
            history.append({'stage':'global','update':step,'validation':score,'elapsed_seconds':time.perf_counter()-begin})
            if score['mean_kl']<best:
                best=score['mean_kl'];beststep=step;save_bank(bank,finalpath)
            dump(reportpath,{'status':'training','window':start,'history':history})
            print('GLOBAL',start,step,round(score['mean_kl'],6),flush=True)
        if step==60:break
        idx=indices[300+step];n=[3,2][step%2];batch=teacher[idx]
        active=install(model,original,{start:{'bank':bank,'steps':n}},capture=True)[start]
        optimizer.zero_grad(set_to_none=True)
        logits=model(windows['train'][idx,:-1],use_cache=False).logits
        with torch.no_grad():lp=F.log_softmax(model.lm_head(traincache['hidden'][idx]),-1)
        kl=(lp.exp()*(lp-F.log_softmax(logits,-1))).sum(-1).mean()
        local=path_loss([b.last_output for b in active],batch,floor)
        loss=kl+.1*local;loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(bank.parameters(),1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
        for b in active:b.last_output=None
        if (step+1)%10==0:print('UPDATE',start,'global',step+1,float(kl.detach()),flush=True)
    restore(model,original);load_bank(bank,finalpath)
    result={'status':'complete','window':start,'history':history,'floor':floor,
        'local_best':localbest,'global_best':{'update':beststep,'mean_validation_kl':best},
        'trainable_slope_values':sum(p.numel() for p in bank.parameters()),
        'elapsed_seconds':time.perf_counter()-begin,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    dump(reportpath,result);print('TRAINED',start,result['elapsed_seconds'],flush=True)


if __name__=='__main__':
    setup();protocol();windows=data();model=load_model();original=list(model.model.layers);before=hashes(model)
    traincache=torch.load(ROOT/'cache/train.pt',weights_only=True);valcache=torch.load(ROOT/'cache/validation.pt',weights_only=True)
    for start in WINDOWS:train_one(model,original,start,traincache,valcache,windows)
    restore(model,original);after=hashes(model);assert before==after
    dump(OUT/'training_integrity.json',{'original_parameter_hashes_unchanged':before==after,'tensors_checked':len(before)})
