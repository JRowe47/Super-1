"""Static recursive Haar update masks with exact per-band counts and step-L2 controls."""
import os
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
import scoring as S
from transforms import haar
from run_retention import dump,sha,lm_eval,replacements,install,copy_batch,kl,bands
ROOT=Path(__file__).resolve().parent
ARMS=['full','targeted','random0','random1','opposite']
def segments(n):
    small=n;levels=0
    while small%2==0:small//=2;levels+=1
    yield 'approximation',levels,0,small
    off=small;length=small
    for level in range(levels,0,-1):
        yield 'detail',level,off,off+length;off+=length;length*=2
    assert off==n
def copy_targets(rows):
    n=rows[0]['n'];targets=[]
    for j,r in enumerate(rows):
        a=np.array(r['prefix']);order=np.arange(n-1) if j==0 else np.array(r['order'])
        targets.append(a[order+1])
    return torch.tensor(np.stack(targets))
def build_masks(m,names,shapes,initial,data,inputs):
    t=time.time();master=[torch.nn.Parameter(haar(v)) for v in initial]
    for i in range(6):
        x=torch.tensor(data['train'][2*i:2*i+2])
        loss=functional_call(m,replacements(names,shapes,master,True),(),dict(input_ids=x,labels=x,use_cache=False)).loss/6
        loss.backward();del loss
    ga=[v.grad.detach().clone() for v in master]
    for v in master:v.grad=None
    for i in range(12):
        rows=inputs['sets']['mask_selection'][2*i:2*i+2];ids,pos=copy_batch(rows);target=copy_targets(rows)
        h=functional_call(m.gpt_neox,replacements(names,shapes,master,True,True),(),dict(input_ids=ids,use_cache=False)).last_hidden_state[:,pos]
        loss=F.cross_entropy(m.embed_out(h).transpose(1,2),target)/12;loss.backward();del h,loss
    gcop=[v.grad.detach().clone() for v in master]
    masks={a:[] for a in ARMS[1:]};rows=[];summary=[];rngs={a:np.random.default_rng(2026101720+j) for j,a in enumerate(['random0','random1'])}
    predicted={a:dict(copy_nll_slope=0.,adaptation_nll_slope=0.,proposal_energy=0.) for a in ARMS}
    for ti,(name,g,q) in enumerate(zip(names,ga,gcop)):
        proposal=(-g/(g.abs()+1e-8)).numpy();g=g.numpy();q=q.numpy();n=len(proposal)
        mm={a:np.ones(n,dtype=bool) for a in masks}
        for kind,level,start,end in segments(n):
            size=end-start
            if size>=128:
                assert size%64==0
                scores=(q[start:end]*proposal[start:end]).reshape(-1,64).sum(1)
                gain=(g[start:end]*proposal[start:end]).reshape(-1,64).sum(1)
                num=len(scores);count=num//2;order=np.argsort(scores,kind='stable')
                selected={'targeted':order[:count],'opposite':order[-count:]}
                for arm in rngs:selected[arm]=rngs[arm].choice(num,count,replace=False)
                for arm in masks:
                    tilemask=np.zeros(num,dtype=bool);tilemask[selected[arm]]=True;mm[arm][start:end]=np.repeat(tilemask,64)
                for j,(score,utility) in enumerate(zip(scores,gain)):
                    rows.append((ti,level,start+64*j,64,float(score),float(utility),*[int(mm[a][start+64*j]) for a in masks]))
            counts={a:int(mm[a][start:end].sum()) for a in masks};assert len(set(counts.values()))==1
            summary.append(dict(tensor=name,layer=int(name.split('.')[2]),kind=kind,level=level,support=2**level,start=start,end=end,count=size,enabled=counts,scored=size>=128))
        for arm in ARMS:
            mask=np.ones(n,dtype=bool) if arm=='full' else mm[arm]
            predicted[arm]['copy_nll_slope']+=float((q*proposal*mask).sum())
            predicted[arm]['adaptation_nll_slope']+=float((g*proposal*mask).sum())
            predicted[arm]['proposal_energy']+=float((proposal*mask)**2 @ np.ones(n))
            if arm!='full':masks[arm].append(torch.from_numpy(mask))
    norm=np.sqrt(predicted['full']['proposal_energy'])
    for arm,p in predicted.items():
        p['norm_match_scale']=norm/np.sqrt(p['proposal_energy'])
        p['predicted_copy_delta_at_lr2e5']=2e-5*p['copy_nll_slope']*p['norm_match_scale']
        p['predicted_adapt_delta_at_lr2e5']=2e-5*p['adaptation_nll_slope']*p['norm_match_scale']
    np.savez_compressed(ROOT/'results/tile_scores.npz',rows=np.array(rows),columns=np.array(['tensor_index','level','start','count','copy_slope','adapt_slope',*masks]))
    for arm,mask in masks.items():np.savez_compressed(ROOT/'models'/f'mask_{arm}.npz',**{n:np.packbits(v.numpy(),bitorder='little') for n,v in zip(names,mask)})
    dump('mask_band_counts.json',summary);dump('mask_calibration.json',dict(predicted=predicted,seconds=time.time()-t,scored_tiles=len(rows),total_coefficients=sum(v.numel() for v in initial),enabled={a:sum(int(v.sum()) for v in mm) for a,mm in masks.items()}))
    print('MASK CALIBRATION',len(rows),'tiles',predicted,flush=True)
    del master,ga,gcop;gc.collect();return masks
def main(checkpoints):
    start=time.time();torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);S.CHECKPOINTS=checkpoints
    protocol_sha=sha(ROOT/'protocol.json');manifest=json.loads((ROOT/'checkpoint_manifest.json').read_text());assert sha(checkpoints/'step143000/model.safetensors')==manifest['sha256']
    z=np.load(ROOT/'data/phase_blocks.npz');data={k:z[k].copy() for k in z.files};inputs=S.read('probe_inputs.json');m=S.model(143000)
    for p in m.parameters():p.requires_grad_(False)
    entries=[(n,p) for n,p in m.named_parameters() if n.startswith('gpt_neox.layers.')];names=[n for n,p in entries];shapes=[tuple(p.shape) for n,p in entries];initial=[p.detach().double().flatten().clone() for n,p in entries]
    assert len(names)==72 and sum(v.numel() for v in initial)==18914304
    dump('tensor_mapping.json',[dict(name=n,shape=s,count=v.numel()) for n,s,v in zip(names,shapes,initial)])
    masks=build_masks(m,names,shapes,initial,data,inputs)
    cache=[];textpos=torch.linspace(0,94,8).round().long()
    with torch.no_grad():
        for i in range(12):
            ids,pos=copy_batch(inputs['sets']['calibration'][2*i:2*i+2]);h=m.gpt_neox(ids,use_cache=False).last_hidden_state[:,pos]
            copyteacher=F.log_softmax(m.embed_out(h).double(),dim=-1)
            textids=torch.tensor(data['replay'][2*i:2*i+2]);h=m.gpt_neox(textids,use_cache=False).last_hidden_state[:,textpos]
            textteacher=F.log_softmax(m.embed_out(h).double(),dim=-1);cache.append((ids,pos,copyteacher,textids,textteacher))
        base_ret,base_hidden,prose_teacher=S.evaluate(m,'retention')
        last_teacher={k:F.log_softmax(m.embed_out(torch.from_numpy(v)).double(),dim=-1) for k,v in base_hidden.items()}
    allrows=[];training=[];bandrows=[];runtimes=[];stepnorms=[];checks=[]
    def evaluate(arm,step):
        with torch.inference_mode():
            lm=lm_eval(m,data['validation']);ret,hidden,_=S.evaluate(m,'retention',prose_teacher)
            for kind in ['repeat','scrambled']:
                lp=F.log_softmax(m.embed_out(torch.from_numpy(hidden[kind+'_hidden'])).double(),dim=-1);old=last_teacher[kind+'_hidden'];vals=(old.exp()*(old-lp)).sum(-1).tolist()
                for row,value in zip(ret[kind]['sequences'],vals):row['last_query_teacher_kl']=value
            row=dict(arm=arm,step=step,validation=lm,retention=ret);allrows.append(row);dump('evaluations.json',allrows)
            short=[r['accuracy'] for r in ret['repeat']['sequences'] if r['n']<=48]
            print('EVAL',arm,step,'NLL',round(lm['nll'],6),'copy',round(float(np.mean(short)),4),flush=True)
    evaluate('baseline',0)
    for arm in ARMS:
        t=time.time();master=[torch.nn.Parameter(haar(v)) for v in initial];frozen=[v.detach().clone() for v in master] if arm!='full' else None
        opt=torch.optim.AdamW(master,lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        install(m,names,shapes,master,True)
        for step in range(1,25):
            opt.zero_grad(set_to_none=True);x=torch.tensor(data['train'][2*(step-1):2*step])
            loss=functional_call(m,replacements(names,shapes,master,True),(),dict(input_ids=x,labels=x,use_cache=False)).loss;av=float(loss.detach());loss.backward();del loss
            ids,pos,teacher,txt,txtteacher=cache[(step-1)%12]
            h=functional_call(m.gpt_neox,replacements(names,shapes,master,True,True),(),dict(input_ids=ids,use_cache=False)).last_hidden_state[:,pos]
            loss=kl(m.embed_out(h),teacher);cv=float(loss.detach());(.5*loss).backward();del h,loss
            h=functional_call(m.gpt_neox,replacements(names,shapes,master,True,True),(),dict(input_ids=txt,use_cache=False)).last_hidden_state[:,textpos]
            loss=kl(m.embed_out(h),txtteacher);tv=float(loss.detach());(.5*loss).backward();del h,loss
            if arm!='full':
                for v,mask in zip(master,masks[arm]):v.grad.mul_(mask)
            before=[v.detach().clone() for v in master];opt.step()
            with torch.no_grad():
                proposal_norm=np.sqrt(sum(float((v-b).square().sum()) for v,b in zip(master,before)))
                if arm=='full':stepnorms.append(proposal_norm);scale=1.
                else:
                    scale=stepnorms[step-1]/proposal_norm
                    for v,b in zip(master,before):v.copy_(b+(v-b)*scale)
                actual_norm=np.sqrt(sum(float((v-b).square().sum()) for v,b in zip(master,before)))
                assert abs(actual_norm-stepnorms[step-1])<1e-10
            del before
            training.append(dict(arm=arm,step=step,adaptation_nll=av,copy_kl=cv,text_kl=tv,proposal_norm=proposal_norm,step_norm=actual_norm,norm_scale=scale));dump('training.json',training)
            if step in [1,12,24]:
                if arm!='full':
                    frozen_error=max(float((v.detach()[~mask]-b[~mask]).abs().max()) if (~mask).any() else 0. for v,b,mask in zip(master,frozen,masks[arm]))
                    assert frozen_error==0
                    checks.append(dict(arm=arm,step=step,frozen_coefficient_max_change=frozen_error))
                install(m,names,shapes,master,True);evaluate(arm,step)
                if step in [12,24]:bandrows.extend(bands(m,names,initial,arm,step));dump('haar_update_bands.json',bandrows)
        if arm in ['full','targeted']:
            arrays={n:dict(m.named_parameters())[n].detach().numpy().copy() for n in names};np.savez_compressed(ROOT/'models'/f'{arm}_final_decoder.npz',**arrays)
        runtimes.append(dict(arm=arm,seconds=time.time()-t,steps=24));print('COMPLETE',arm,round(time.time()-t,2),flush=True)
        del opt,master,frozen;gc.collect()
    dump('mask_checks.json',checks);dump('reference_step_norms.json',stepnorms)
    assert sha(ROOT/'protocol.json')==protocol_sha
    dump('environment.json',dict(seconds=time.time()-start,runs=runtimes,new_training_steps=120,threads=3,torch=torch.__version__,protocol_sha256=protocol_sha,checkpoint_sha256=manifest['sha256']))
    print('MASK EXPERIMENT COMPLETE',round(time.time()-start,2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,required=True);a=p.parse_args();main(a.checkpoints)
