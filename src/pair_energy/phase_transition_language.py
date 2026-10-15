"""Local WikiText byte pilot: state-conditioned rotation transition ablation."""
import argparse,copy,hashlib,json,math,time,statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import capacity_sweep as shared

C={'seeds':[2503,2551,2609],'lrs':[.001,.003],'steps':300,'batch':8,'context':64,
   'width':64,'validation_interval':50,'validation_windows':64,'validation_start':0,
   'arms':['fixed_rotation','geometric_rotation','transformer'],'threads':1}
DATA=Path('/Volumes/SP AI 01_16/w560-function-weight-wt2-word-sota-20260828/data/wikitext-2')

class PhaseLM(nn.Module):
    def __init__(self,d,dynamic):
        super().__init__();self.dynamic=dynamic;self.d=d
        self.embedding=nn.Embedding(256,d);nn.init.normal_(self.embedding.weight,std=.02)
        self.drive=nn.Linear(d,d);self.write=nn.Linear(d,d);self.retention=nn.Parameter(torch.zeros(d))
        self.angle=nn.Parameter(torch.zeros(d//2));self.gain=nn.Parameter(torch.zeros(d//2)) if dynamic else None
        self.norm=nn.LayerNorm(d)
    def forward(self,tokens):
        x=self.embedding(tokens);drive=torch.tanh(self.drive(x));g=torch.sigmoid(self.write(x)+self.retention)
        h=x.new_zeros((len(x),self.d));history=[]
        for i in range(tokens.shape[1]):
            pair=h.reshape(len(h),-1,2);u=drive[:,i].reshape(len(h),-1,2)
            theta=self.angle
            if self.dynamic:
                cross=pair[:,:,0]*u[:,:,1]-pair[:,:,1]*u[:,:,0]
                denom=torch.sqrt(pair.square().sum(-1)*u.square().sum(-1)+1e-6)
                theta=theta+self.gain*torch.tanh(cross/denom)
            co=torch.cos(theta);si=torch.sin(theta)
            rotated=torch.stack((co*pair[:,:,0]-si*pair[:,:,1],si*pair[:,:,0]+co*pair[:,:,1]),-1).flatten(1)
            h=g[:,i]*rotated+(1-g[:,i])*drive[:,i];history.append(h)
        return self.norm(torch.stack(history,1))@self.embedding.weight.T

class TinyTransformer(nn.Module):
    def __init__(self,d,ff=None):
        super().__init__();self.embedding=nn.Embedding(256,d);nn.init.normal_(self.embedding.weight,std=.02)
        self.position=nn.Parameter(torch.randn(C['context'],d)*.02)
        self.block=nn.TransformerEncoderLayer(d,2,ff if ff is not None else 2*d,dropout=0.,batch_first=True,norm_first=True)
        self.norm=nn.LayerNorm(d)
    def forward(self,tokens):
        x=self.embedding(tokens)+self.position[:tokens.shape[1]]
        mask=torch.ones(tokens.shape[1],tokens.shape[1],device=x.device,dtype=torch.bool).triu(1)
        return self.norm(self.block(x,src_mask=mask))@self.embedding.weight.T

def count(m):return sum(p.numel() for p in m.parameters())
def evaluate(m,data):
    losses=[]
    with torch.inference_mode():
        for start in range(0,C['validation_windows'],C['batch']):
            ids=torch.arange(start,min(start+C['batch'],C['validation_windows']))*C['context']+C['validation_start']
            x=torch.stack([data[i:i+C['context']] for i in ids]);y=torch.stack([data[i+1:i+C['context']+1] for i in ids])
            losses.append((F.cross_entropy(m(x).reshape(-1,256),y.flatten(),reduction='sum').item(),y.numel()))
    ce=sum(a for a,b in losses)/sum(b for a,b in losses);return {'ce':ce,'bpc':ce/math.log(2),'scored_bytes':sum(b for a,b in losses)}

def checks():
    torch.manual_seed(0);a=PhaseLM(16,False).double();b=PhaseLM(16,True).double();b.load_state_dict(a.state_dict(),strict=False)
    x=torch.randint(256,(2,8));assert torch.allclose(a(x),b(x),atol=1e-11,rtol=1e-10)
    with torch.no_grad():b.gain.fill_(.7)
    before=b(x);changed=x.clone();changed[:,4:]=torch.randint(256,(2,4));assert torch.equal(before[:,:4],b(changed)[:,:4])
    before.square().sum().backward();assert torch.isfinite(b.gain.grad).all() and b.gain.grad.abs().sum()>0
    pair=torch.randn(7,2,dtype=torch.float64);angle=torch.randn(7,dtype=torch.float64)
    rot=torch.stack((angle.cos()*pair[:,0]-angle.sin()*pair[:,1],angle.sin()*pair[:,0]+angle.cos()*pair[:,1]),-1)
    assert torch.allclose(pair.square().sum(1),rot.square().sum(1),atol=1e-12)
    return {'zero_gain_identity':True,'causal_prefix':True,'geometry_gradient_nonzero':True,'rotation_norm_preserving':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--canary',action='store_true');args=p.parse_args()
    torch.set_num_threads(C['threads']);torch.use_deterministic_algorithms(True);proof=checks()
    if args.canary:
        m=PhaseLM(C['width'],True);x=torch.randint(256,(C['batch'],C['context']));o=torch.optim.AdamW(m.parameters());t=time.perf_counter()
        for _ in range(10):o.zero_grad();F.cross_entropy(m(x).reshape(-1,256),x.flatten()).backward();o.step()
        print({'checks':proof,'estimated_candidate_seconds':(time.perf_counter()-t)*C['steps']/10});return
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    raw={name:(DATA/(name+'.txt')).read_bytes() for name in ['train','valid']}
    train=torch.tensor(list(raw['train']),dtype=torch.long);valid=torch.tensor(list(raw['valid']),dtype=torch.long)
    target=count(PhaseLM(C['width'],True));d=min(range(16,80,2),key=lambda v:abs(count(TinyTransformer(v))-target))
    ff=min(range(d,4*d+1),key=lambda v:abs(count(TinyTransformer(d,v))-target))
    builders={'fixed_rotation':lambda:PhaseLM(C['width'],False),'geometric_rotation':lambda:PhaseLM(C['width'],True),'transformer':lambda:TinyTransformer(d,ff)}
    counts={arm:count(make()) for arm,make in builders.items()};assert max(counts.values())/min(counts.values())<1.02
    shared.write(out/'PROTOCOL.json',{'config':C,'checks':proof,'parameters':counts,'transformer_width':d,'transformer_ff_width':ff,'source_sha256':shared.prior.sha(Path(__file__)),'data_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in raw.items()},
        'equation':'Angle = learned constant + learned gain * tanh(normalized oriented area between old state pair and current drive pair). Per-pair rotation generates the recurrent transition, followed by input-dependent write/retention.',
        'scope':'New exploratory successor, not a reimplementation of original hierarchy. Unitary/rotational RNN prior work exists. Byte-level WikiText validation subset, not word-level perplexity or official test.',
        'contract':'Each arm receives identical training byte windows per seed, equal updates and two LR trials, parameter counts within 2%; reset state at every window for equal context. Equal FLOPs are not claimed. Validation-only selection. Official test never read.'})
    rows=[];selected=[]
    for seed in C['seeds']:
        for arm,make in builders.items():
            candidates=[]
            for lr in C['lrs']:
                torch.manual_seed(seed);m=make();o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.01);rng=torch.Generator().manual_seed(seed+10000);best=float('inf');history=[];start=time.perf_counter()
                for step in range(1,C['steps']+1):
                    ids=torch.randint(len(train)-C['context']-1,(C['batch'],),generator=rng)
                    x=torch.stack([train[i:i+C['context']] for i in ids]);y=torch.stack([train[i+1:i+C['context']+1] for i in ids])
                    for group in o.param_groups:group['lr']=lr*(.1+.9*.5*(1+math.cos(math.pi*step/C['steps'])))
                    o.zero_grad();loss=F.cross_entropy(m(x).reshape(-1,256),y.flatten());assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
                    if step%C['validation_interval']==0:
                        score=evaluate(m,valid);history.append({'step':step,**score})
                        if score['ce']<best:best=score['ce'];state=copy.deepcopy(m.state_dict());best_step=step
                m.load_state_dict(state);path=out/f'{seed}-{arm}-{lr}.pt';torch.save(state,path)
                r={'seed':seed,'arm':arm,'lr':lr,'validation':evaluate(m,valid),'best_step':best_step,'parameters':count(m),'history':history,'training_seconds':time.perf_counter()-start,'checkpoint':str(path.resolve()),'sha256':shared.prior.sha(path)}
                rows.append(r);candidates.append(r);shared.write(out/'CANDIDATES.json',rows);print(json.dumps({'completed':len(rows),'total':18,'arm':arm,'validation':r['validation']}),flush=True)
            selected.append(min(candidates,key=lambda r:r['validation']['ce']))
    shared.write(out/'FREEZE.json',selected)
    summary=[]
    for arm in builders:
        rs=[r for r in selected if r['arm']==arm];summary.append({'arm':arm,'mean_validation_ce':statistics.mean(r['validation']['ce'] for r in rs),'mean_validation_bpc':statistics.mean(r['validation']['bpc'] for r in rs),'parameters':counts[arm]})
    shared.write(out/'SUMMARY.json',summary);shared.write(out/'COMPLETION.json',{'candidates':18,'official_test_read':False,'cloud_resources_created':0})
    lines=['# Geometric recurrent-transition language pilot','','Validation-only WikiText-2 byte model, three seeds, 300 updates, two learning rates. This is an exploratory local screen. Every training window and context reset is matched; compute and runtime are not matched. The official test is sealed.','','| Model | Parameters | Validation CE | BPC |','|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['arm']} | {r['parameters']} | {r['mean_validation_ce']:.5f} | {r['mean_validation_bpc']:.5f} |")
    lines+=['','Only the first 4,096 nonoverlapping validation target bytes are scored, with window resets. All architectures and LRs use that same subset. This is not exhaustive validation, independent confirmation, word-level perplexity, novelty evidence, or SOTA. More training and larger independent validation are required for a promising result.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
