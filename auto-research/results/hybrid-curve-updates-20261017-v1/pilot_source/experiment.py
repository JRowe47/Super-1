"""Full Fourier coordinates and selectively predicted updates: synthetic probe.

New two-layer model, NOT a substitute for the historical GRU/LM architecture.
Composition uses an explicit common executor; parsing is not learned here.
"""
import argparse, copy, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
ARMS = ['backprop', 'hybrid', 'gated_skip', 'always_hyper']

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encode(w):
    """N real orthonormal coordinates, including sqrt(2) interior scaling."""
    w = w.reshape(-1); n = w.numel()
    z = torch.fft.rfft(w, norm='ortho')
    middle = z[1:-1] if n % 2 == 0 else z[1:]
    out = [z[:1].real, torch.view_as_real(middle).reshape(-1) * math.sqrt(2)]
    if n % 2 == 0: out.append(z[-1:].real)
    return torch.cat(out)

def decode(c):
    n = c.numel(); k = (n-1)//2
    middle = (c[1:1+2*k] / math.sqrt(2)).reshape(k, 2).contiguous()
    out = [torch.complex(c[:1], torch.zeros_like(c[:1])), torch.view_as_complex(middle)]
    if n % 2 == 0: out.append(torch.complex(c[-1:], torch.zeros_like(c[-1:])))
    return torch.fft.irfft(torch.cat(out), n=n, norm='ortho')

class Model(nn.Module):
    def __init__(self, width=32):
        super().__init__(); self.width=width
        a=torch.randn(width,17)/4; a[:,-1]=0
        b=torch.randn(8,width+1)/math.sqrt(width); b[:,-1]=0
        self.c1=nn.Parameter(encode(a)); self.c2=nn.Parameter(encode(b))
    def forward(self,x,cs=None):
        c1,c2=(self.c1,self.c2) if cs is None else cs
        w1=decode(c1).reshape(self.width,17);w2=decode(c2).reshape(8,self.width+1)
        z=F.linear(x,w1[:,:-1],w1[:,-1]);h=z.tanh()
        logits=F.linear(h,w2[:,:-1],w2[:,-1])
        return logits,h,z,w1,w2
    @torch.no_grad()
    def apply(self,delta):
        self.c1.add_(delta[0]);self.c2.add_(delta[1])

class UpdateNet(nn.Module):
    def __init__(self,lr):
        super().__init__();self.lr=lr
        self.hidden=nn.Sequential(nn.Linear(25,32),nn.Tanh(),nn.Linear(32,1))
        self.output=nn.Sequential(nn.Linear(2,8),nn.Tanh(),nn.Linear(8,1))
        for network in [self.hidden,self.output]:
            nn.init.zeros_(network[-1].weight);nn.init.zeros_(network[-1].bias)
    def forward(self,x,h,w2,prob,y):
        b,d=h.shape; yy=F.one_hot(y,8).to(h)
        # Only forward activations, current weights, predictions and observed
        # labels enter the predictor. No autograd derivatives enter its inputs.
        features=torch.cat([h[:,:,None],w2[:,:-1].T[None].expand(b,-1,-1),
            prob[:,None,:].expand(-1,d,-1),yy[:,None,:].expand(-1,d,-1)],-1)
        r1=2*self.hidden(features).squeeze(-1).tanh()
        r2=self.lr*self.output(torch.stack([prob,yy],-1)).squeeze(-1).tanh()
        # Learned factors assemble complete affine-layer parameter deltas.
        one=torch.ones(b,1,dtype=x.dtype)
        dw1=r1.T@torch.cat([x,one],1)/b
        dw2=r2.T@torch.cat([h,one],1)/b
        delta=(encode(dw1),encode(dw2))
        norm=torch.sqrt(sum(t.square().sum() for t in delta))
        scale=(.5/norm.clamp_min(1e-12)).clamp_max(1)
        return tuple(t*scale for t in delta),r1,r2

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(10,24),nn.Tanh(),nn.Linear(24,3))
    def forward(self,x): return self.net(x)

def world(seed):
    rng=np.random.default_rng(seed+1700000)
    return np.stack([rng.permutation(8) for _ in range(8)])

def batch(table, seed, ops, n=64):
    rng=np.random.default_rng(seed);a=rng.choice(ops,n);v=rng.integers(0,8,n)
    x=torch.cat([F.one_hot(torch.tensor(v),8),F.one_hot(torch.tensor(a),8)],1).float()
    y=torch.tensor(table[a,v]);return x,y

def episode(table,seed,step,phase,n=64):
    key=seed*100000+step*10
    if phase<0: train=batch(table,key,list(range(4)),n); newops=list(range(4))
    else:
        op=phase+4;fresh=batch(table,key,[op],n//2)
        old=batch(table,key+1,list(range(op)),n//2)
        train=tuple(torch.cat([a,b]) for a,b in zip(fresh,old));newops=[op]
    return train,batch(table,key+2,newops,n),batch(table,key+3,list(range(4)),n)

def teacher(model,x,y,lr):
    logits,h,z,w1,w2=model(x);loss=F.cross_entropy(logits,y)
    g1,g2,gz,gl=torch.autograd.grad(loss,[model.c1,model.c2,z,logits])
    return (-lr*g1,-lr*g2),(-lr*len(x)*gz,-lr*len(x)*gl),tuple(t.detach() for t in [logits,h,w1,w2]),float(loss.detach())

def features(pack,delta,y,memory):
    logits,h,w1,w2=pack;p=logits.softmax(-1)
    loss=F.cross_entropy(logits,y);entropy=-(p*p.clamp_min(1e-9).log()).sum(-1).mean()
    vals=[loss/3,entropy/3,h.square().mean(),w1.square().mean(),w2.square().mean(),
          delta[0].norm(),delta[1].norm(),(logits.argmax(-1)==y).float().mean(),
          loss.new_tensor(memory[0]),loss.new_tensor(memory[1])]
    return torch.stack(vals).detach()

@torch.no_grad()
def effects(model,pred,ref,queries):
    before=[];candidate=[];reference=[];accuracies=[]
    for x,y in queries:
        z=model(x)[0];p=model(x,(model.c1+pred[0],model.c2+pred[1]))[0]
        t=model(x,(model.c1+ref[0],model.c2+ref[1]))[0]
        before.append(float(F.cross_entropy(z,y)));candidate.append(float(F.cross_entropy(p,y)))
        reference.append(float(F.cross_entropy(t,y)));accuracies.append(float((p.argmax(-1)==y).float().mean()))
    relerr=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(pred,ref)))/
                 torch.sqrt(sum(t.square().sum() for t in ref)).clamp_min(1e-8))
    return dict(safe=all(a<=b+.01 for a,b in zip(candidate,reference)),
        gain=float(np.mean(candidate)-np.mean(before)),state_accuracy=float(np.mean(accuracies)),
        candidate_loss=candidate,reference_loss=reference,relative_update_error=relerr)

def memory_update(memory,effect,pred_gain):
    return [.95*memory[0]+.05*min(effect['relative_update_error']/10,1),
            .95*memory[1]+.05*min(abs(effect['gain']-pred_gain),1)]

def wilson_lower(success,total,z=1.6448536269514722):
    if not total:return 0.
    p=success/total
    return (p+z*z/(2*total)-z*math.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total)

def calibrate(rows):
    trials=[];chosen=None
    for threshold in [.5,.7,.8,.9,.95,.99]:
        a=[r for r in rows if r['confidence']>=threshold]
        good=sum(r['safe'] for r in a);lower=wilson_lower(good,len(a))
        record=dict(threshold=threshold,count=len(a),safe=good,lower95=lower);trials.append(record)
        if chosen is None and len(a)>=64 and lower>=.95:chosen=threshold
    return dict(threshold=chosen,loss_limit=max(r['train_loss'] for r in rows),
        qualification=chosen is not None,trials=trials,
        note='Wilson bound is a heuristic qualification on temporally related calibration steps, not an IID guarantee.')

def warmup(seed,cfg,out):
    table=world(seed);torch.manual_seed(seed);model=Model(cfg['width'])
    predictor=UpdateNet(cfg['lr']);gate=Gate()
    oh=torch.optim.Adam(predictor.parameters(),lr=.001)
    og=torch.optim.Adam(gate.parameters(),lr=.001)
    memory=[0.,0.];rows=[];trace=[];start=time.process_time();wall=time.perf_counter()
    total=cfg['warmup']+cfg['calibration']
    for step in range(total):
        (x,y),qn,qo=episode(table,seed,step,-1,cfg['batch'])
        ref,targets,pack,loss=teacher(model,x,y,cfg['lr'])
        logits,h,w1,w2=pack
        pred,r1,r2=predictor(x,h,w2,logits.softmax(-1),y)
        f=features(pack,pred,y,memory);g=gate(f)
        effect=effects(model,tuple(t.detach() for t in pred),ref,[qn,qo])
        if step<cfg['warmup']:
            oh.zero_grad();hloss=F.mse_loss(r1/.2,targets[0]/.2)+F.mse_loss(r2/cfg['lr'],targets[1]/cfg['lr'])
            hloss.backward();oh.step()
            target=g.new_tensor([float(effect['safe']),effect['state_accuracy'],effect['gain']])
            gloss=F.binary_cross_entropy_with_logits(g[:2],target[:2])+10*(g[2]-target[2]).square()
            og.zero_grad();gloss.backward();og.step()
        else:
            rows.append(dict(step=step,confidence=float(g[0].sigmoid().detach()),
                state_confidence=float(g[1].sigmoid().detach()),predicted_gain=float(g[2].detach()),train_loss=loss,**effect))
        memory=memory_update(memory,effect,float(g[2].detach()))
        model.apply(ref)
        if step%100==0 or step==total-1:
            trace.append(dict(step=step,loss=loss,confidence=float(g[0].sigmoid().detach()),**effect))
    trained_cpu=time.process_time()-start;trained_wall=time.perf_counter()-wall
    qualification=calibrate(rows)
    torch.save(dict(model=model.state_dict(),predictor=predictor.state_dict(),gate=gate.state_dict(),
        memory=memory,world=table.tolist(),config=cfg),out/'warmup.pt')
    save(out/'warmup_trace.json',trace);save(out/'calibration.json',dict(rule=qualification,rows=rows))
    # Replay plain backprop warmup for an actual startup-cost reference.
    torch.manual_seed(seed);plain=Model(cfg['width']);start=time.process_time();wall=time.perf_counter()
    for step in range(total):
        (x,y),_,_=episode(table,seed,step,-1,cfg['batch'])
        ref,_,_,_=teacher(plain,x,y,cfg['lr']);plain.apply(ref)
    plain_cpu=time.process_time()-start;plain_wall=time.perf_counter()-wall
    parity=max(float((a-b).abs().max().detach()) for a,b in zip(model.parameters(),plain.parameters()))
    assert parity<=cfg['fidelity_fp32'],('warmup_replay_failed',parity)
    result=dict(plain_cpu=plain_cpu,hybrid_cpu=trained_cpu,plain_wall=plain_wall,hybrid_wall=trained_wall,
        base_backward_steps=total,controller_training_steps=cfg['warmup'],warmup_parameter_replay_error=parity,
        model_parameters=sum(p.numel() for p in model.parameters()),
        predictor_parameters=sum(p.numel() for p in predictor.parameters()),gate_parameters=sum(p.numel() for p in gate.parameters()),
        qualification=qualification)
    save(out/'warmup_summary.json',result)
    return model,predictor,gate,memory,table,result

def run(seed,cfg,out):
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();cpu=time.process_time()
    model,predictor,gate,initial_memory,table,warm=warmup(seed,cfg,out)
    for p in predictor.parameters():p.requires_grad_(False)
    for p in gate.parameters():p.requires_grad_(False)
    originals=copy.deepcopy(model.state_dict());summaries={}
    for arm in ARMS:
        model.load_state_dict(originals);memory=list(initial_memory)
        threshold=warm['qualification']['threshold'];limit=warm['qualification']['loss_limit']
        counts=dict(backward=0,proposed=0,eligible=0,applied=0,skipped=0,audits=0,audit_failures=0)
        rows=[];digest=hashlib.sha256();t0=time.process_time();w0=time.perf_counter()
        for step in range(4*cfg['phase_steps']):
            phase=step//cfg['phase_steps'];(x,y),qn,qo=episode(table,seed,10000+step,phase,cfg['batch'])
            digest.update(x.numpy().tobytes());digest.update(y.numpy().tobytes())
            if arm=='backprop':
                ref,_,_,loss=teacher(model,x,y,cfg['lr']);model.apply(ref);counts['backward']+=1
                confidence=None;pred_gain=None;eligible=False;applied=False;audit=None
            else:
                with torch.no_grad():
                    logits,h,_,w1,w2=model(x);pack=(logits,h,w1,w2)
                    pred,_,_=predictor(x,h,w2,logits.softmax(-1),y)
                    g=gate(features(pack,pred,y,memory));confidence=float(g[0].sigmoid());pred_gain=float(g[2])
                    loss=float(F.cross_entropy(logits,y))
                    eligible=(threshold is not None and confidence>=threshold and loss<=limit)
                    if arm=='always_hyper':eligible=True
                counts['proposed']+=1;counts['eligible']+=int(eligible);audit=None;ref=None
                # Shared deterministic random audit mask, independent of data and model.
                audited=eligible and np.random.default_rng(seed*1000000+step).random()<.1
                if audited:
                    ref,_,_,_=teacher(model,x,y,cfg['lr']);counts['backward']+=1
                    audit=effects(model,pred,ref,[qn,qo]);counts['audits']+=1
                    counts['audit_failures']+=int(not audit['safe'])
                    memory=memory_update(memory,audit,pred_gain)
                applied=eligible and (audit is None or audit['safe'] or arm=='always_hyper')
                if applied:
                    if arm=='gated_skip':counts['skipped']+=1
                    else:model.apply(pred);counts['applied']+=1
                else:
                    if ref is None:ref,_,_,_=teacher(model,x,y,cfg['lr']);counts['backward']+=1
                    model.apply(ref)
            assert math.isfinite(loss) and all(torch.isfinite(p).all() for p in model.parameters())
            if step%25==0 or audit is not None or step==4*cfg['phase_steps']-1:
                rows.append(dict(step=step,phase=phase,loss=loss,confidence=confidence,predicted_gain=pred_gain,
                    eligible=eligible,applied=applied,audit=audit,memory=list(memory),counts=dict(counts)))
        elapsed=time.process_time()-t0;wall=time.perf_counter()-w0
        filename=out/f'{arm}.pt'
        torch.save(dict(model=model.state_dict(),world=table.tolist(),seed=seed,arm=arm,config=cfg),filename)
        result=dict(arm=arm,seed=seed,main_cpu=elapsed,main_wall=wall,
            lifecycle_cpu=elapsed+warm['plain_cpu' if arm=='backprop' else 'hybrid_cpu'],
            counts=counts,training_examples=4*cfg['phase_steps']*cfg['batch'],data_sha256=digest.hexdigest(),
            checkpoint_sha256=sha(filename),controller_frozen_after_warmup=True)
        save(out/f'{arm}_trace.json',rows);save(out/f'{arm}_summary.json',result);summaries[arm]=result
        print(json.dumps(dict(seed=seed,arm=arm,main_cpu=elapsed,counts=counts)),flush=True)
    assert len({r['data_sha256'] for r in summaries.values()})==1
    save(out/'TERMINAL.json',dict(status='COMPLETE',seed=seed,wall_seconds=time.perf_counter()-start,
        cpu_seconds=time.process_time()-cpu,warmup=warm,arms=summaries))

@torch.no_grad()
def evaluate(path):
    state=torch.load(path,weights_only=True);model=Model(state['config']['width']);model.load_state_dict(state['model'])
    table=np.array(state['world']);atomic=[];ce=[]
    for op in range(8):
        v=torch.arange(8);x=torch.cat([F.one_hot(v,8),F.one_hot(torch.full_like(v,op),8)],1).float()
        logits=model(x)[0];labels=torch.tensor(table[op]);atomic.append(float((logits.argmax(-1)==labels).float().mean()))
        ce.append(float(F.cross_entropy(logits,labels)))
    rng=np.random.default_rng(3300000+state['seed']);composition={};dh=hashlib.sha256()
    for length in [2,4]:
        if length==2:
            chains=np.array([[a,b] for a in range(8) for b in range(8) for _ in range(8)])
            start=np.tile(np.arange(8),64)
        else:chains=rng.integers(0,8,(2048,4));start=rng.integers(0,8,2048)
        pred=torch.tensor(start);truth=start.copy();dh.update(chains.tobytes());dh.update(start.tobytes())
        for a in chains.T:
            x=torch.cat([F.one_hot(pred,8),F.one_hot(torch.tensor(a),8)],1).float()
            pred=model(x)[0].argmax(-1);truth=table[a,truth]
        composition[str(length)]=float((pred==torch.tensor(truth)).float().mean())
    return dict(atomic_per_operator=atomic,atomic_mean=float(np.mean(atomic)),
        old_retention=float(np.mean(atomic[:4])),new_atomic=float(np.mean(atomic[4:])),
        old_ce=float(np.mean(ce[:4])),new_ce=float(np.mean(ce[4:])),composition=composition,
        confirmation_input_sha256=dh.hexdigest())

def verify():
    rows=[]
    for dtype,tol in [(torch.float32,2e-6),(torch.float64,2e-12)]:
        for n in [1,2,7,16,31,264,544]:
            torch.manual_seed(810+n);w=torch.randn(n,dtype=dtype,requires_grad=True)
            c=encode(w);back=decode(c)
            err=float((w-back).abs().max().detach());normerr=float(abs(w.norm()-c.norm()).detach())
            # Relative norm comparison avoids dimensional scaling of an absolute criterion.
            normrel=normerr/max(float(w.norm().detach()),1)
            assert err<=tol and normrel<=tol,(dtype,n,err,normrel)
            rows.append(dict(dtype=str(dtype),length=n,reconstruction=err,norm_relative_error=normrel,tolerance=tol))
        torch.manual_seed(816);model=Model().to(dtype)
        a=decode(model.c1).detach().reshape(32,17).requires_grad_();b=decode(model.c2).detach().reshape(8,33).requires_grad_()
        x=torch.randn(9,16,dtype=dtype,requires_grad=True);xx=x.detach().clone().requires_grad_();y=torch.arange(9)%8
        logits=model(x)[0];reference=F.linear(F.linear(xx,a[:,:-1],a[:,-1]).tanh(),b[:,:-1],b[:,-1])
        g=torch.autograd.grad(F.cross_entropy(logits,y),(model.c1,model.c2,x))
        gg=torch.autograd.grad(F.cross_entropy(reference,y),(a,b,xx))
        checks=dict(output=float((logits-reference).abs().max().detach()),
            input_gradient=float((g[2]-gg[2]).abs().max()),
            parameter_gradient=max(float((decode(g[i])-gg[i].reshape(-1)).abs().max()) for i in [0,1]))
        for i,c in enumerate([model.c1,model.c2]):
            checks[f'sgd_{i}']=float((decode(c-.5*g[i])-( [a,b][i].reshape(-1)-.5*gg[i].reshape(-1))).abs().max().detach())
        assert max(checks.values())<=tol,(dtype,checks)
        rows.append(dict(dtype=str(dtype),model_checks=checks,tolerance=tol))
    # Verify update assembly against true autograd deltas, using nonzero factors.
    torch.manual_seed(891);m=Model();x,y=batch(world(891),13,list(range(8)))
    ref,factors,pack,_=teacher(m,x,y,.5);h=pack[1];one=torch.ones(len(x),1)
    assembled=[encode(factors[0].T@torch.cat([x,one],1)/len(x)),encode(factors[1].T@torch.cat([h,one],1)/len(x))]
    error=max(float((a-b).abs().max()) for a,b in zip(ref,assembled));assert error<=2e-6,error
    # Application must work without autograd and must actually change parameters.
    net=UpdateNet(.5)
    with torch.no_grad():
        net.hidden[-1].weight.normal_(0,.05);net.output[-1].weight.normal_(0,.05)
        delta,_,_=net(x,h,pack[3],pack[0].softmax(-1),y)
        old=m.c1.clone();m.apply(delta)
        assert not any(t.requires_grad for t in delta) and float((m.c1-old).abs().max())>0
    result=dict(status='PASS',checks=rows,teacher_factor_assembly_max_abs=error,
        no_autograd_application=True,source_sha256=sha(__file__))
    save(ROOT/'VERIFICATION.json',result);print(json.dumps(result,indent=2))

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['verify','pilot','main','evaluate']);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.mode=='verify':verify();return
    p=json.loads((ROOT/('PILOT_PROTOCOL.json' if args.mode=='pilot' else 'PROTOCOL.json')).read_text())
    assert p['source_sha256']==sha(__file__),'Source changed after registration'
    if args.mode in ['pilot','main']:
        assert json.loads((ROOT/'VERIFICATION.json').read_text())['status']=='PASS'
        assert args.seed in p['seeds'];run(args.seed,p['config'],ROOT/args.mode/f's{args.seed}')
    else:
        rows=[]
        for seed in p['seeds']:
            for arm in ARMS:
                path=ROOT/'main'/f's{seed}'/f'{arm}.pt'
                expected=json.loads(path.with_name(f'{arm}_summary.json').read_text())['checkpoint_sha256']
                assert sha(path)==expected
                a=evaluate(path);b=evaluate(path);assert a==b
                row=dict(seed=seed,arm=arm,**a);rows.append(row);print(json.dumps(row),flush=True)
        save(ROOT/'CONFIRMATION.json',dict(rows=rows,exact_replays=True))

if __name__=='__main__':main()
