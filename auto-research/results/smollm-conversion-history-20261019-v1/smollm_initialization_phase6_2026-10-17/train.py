"""Longer matched distillation; independent arm workers can run concurrently."""
import argparse,gc,json,math,platform,resource,time
import numpy as np
import torch
from filelock import FileLock
from torch.nn import functional as F
from common import ROOT,OUT,ARMS,SEED,UPDATES,PEAK_LR,dump,setup,load_model,data,hashes
from models import candidate,install,restore,save_blocks
from evaluation import evaluate

def train_arm(kind):
    with FileLock(str(ROOT/'cache'/f'train_{kind}.lock')):
        run(kind)

def run(kind):
    record=OUT/f'training_{kind}.json';bestpath=ROOT/'checkpoints'/f'{kind}.safetensors'
    if record.exists() and json.loads(record.read_text()).get('status')=='complete':
        print('ALREADY_COMPLETE',kind,flush=True);return
    windows=data();trainref=torch.load(ROOT/'cache/train.pt',weights_only=True)
    valref=torch.load(ROOT/'cache/validation.pt',weights_only=True)
    indices=torch.tensor(np.load(ROOT/'data/training_indices.npy'),dtype=torch.long)
    model=load_model();original=list(model.model.layers);before=hashes(model)
    blocks=candidate(original,kind,trainable=True);install(model,original,blocks)
    optimizer=torch.optim.AdamW(blocks.parameters(),lr=PEAK_LR,weight_decay=0.)
    resume_path=ROOT/'cache'/f'{kind}_resume.pt';history=[];loss_rows=[];best=float('inf');beststep=0;first=0;prior_seconds=0.
    if resume_path.exists():
        state=torch.load(resume_path,weights_only=True)
        blocks.load_state_dict(state['weights']);optimizer.load_state_dict(state['optimizer'])
        first=state['step'];history=state['history'];loss_rows=state['loss_batches']
        best=state['best'];beststep=state['best_step'];prior_seconds=state['elapsed_seconds'];del state
    begin=time.perf_counter()
    for step in range(first,UPDATES+1):
        if step%300==0 and not any(r['update']==step for r in history):
            metrics=evaluate(model,windows['validation'][:8],valref)
            compact={k:metrics[k] for k in ['mean_kl','delta_nll','perplexity_ratio_to_original']}
            history.append({'update':step,'validation':compact,'elapsed_seconds':prior_seconds+time.perf_counter()-begin})
            if metrics['mean_kl']<best:
                best=metrics['mean_kl'];beststep=step;save_blocks(blocks,bestpath)
            dump(record,{'status':'training','arm':kind,'history':history,'loss_batches':loss_rows,'best_update':beststep})
            print('VALIDATE',kind,step,compact,flush=True)
            state={'weights':blocks.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                'history':history,'loss_batches':loss_rows,'best':best,'best_step':beststep,
                'elapsed_seconds':prior_seconds+time.perf_counter()-begin}
            temp=resume_path.with_suffix('.tmp');torch.save(state,temp);temp.replace(resume_path);del state
        if step==UPDATES:break
        scale=(step+1)/60 if step<60 else .2+.8*.5*(1+math.cos(math.pi*(step-60)/(UPDATES-60)))
        for group in optimizer.param_groups:group['lr']=PEAK_LR*scale
        idx=indices[step];optimizer.zero_grad(set_to_none=True)
        logits=model(windows['train'][idx,:-1],use_cache=False).logits
        with torch.no_grad():lp=F.log_softmax(model.lm_head(trainref['hidden'][idx]),-1)
        loss=(lp.exp()*(lp-F.log_softmax(logits,-1))).sum(-1).mean()
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(blocks.parameters(),1.)
        assert torch.isfinite(loss) and torch.isfinite(grad),(kind,step)
        optimizer.step()
        if (step+1)%30==0:
            row={'update':step+1,'teacher_kl':float(loss.detach()),'lr':PEAK_LR*scale,'preclip_grad_norm':float(grad)}
            loss_rows.append(row);print('UPDATE',kind,row,flush=True)
    restore(model,original);assert before==hashes(model)
    result={'status':'complete','arm':kind,'history':history,'loss_batches':loss_rows,
        'best_update':beststep,'best_mean_validation_kl':best,'peak_lr':PEAK_LR,
        'trainable_values':sum(p.numel() for p in blocks.parameters()),'updates':UPDATES,
        'input_tokens':UPDATES*256,'elapsed_seconds':prior_seconds+time.perf_counter()-begin,
        'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'original_parameter_hashes_unchanged':True,'seed':SEED}
    dump(record,result);dump(OUT/f'integrity_{kind}.json',{'all_original_parameters_unchanged':True,'parameter_sha256':before})
    print('TRAINING_COMPLETE',kind,result['elapsed_seconds'],flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS+['all'],default='all');args=parser.parse_args()
    setup()
    for kind in ARMS if args.arm=='all' else [args.arm]:train_arm(kind);gc.collect()

if __name__=='__main__':main()
