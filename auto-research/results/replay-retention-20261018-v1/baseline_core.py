"""SAM-inspired diagnostic, independently implemented from arXiv v2 Algorithm 2.

All parameters are complete real orthonormal Fourier stores, per matrix.
Local rules are evaluated in dense synapse coordinates then transformed.
The local path has no autograd or access to evaluation information.
"""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F

D, H, K, R = 11, 32, 64, 4
SHAPES = {'W0': (H,D), 'W1': (H,H), 'W2': (R,H),
          'B1': (K,K), 'B2': (K,R), 'V1': (K,H), 'V2': (K,H),
          'U0': (H,K), 'U1': (H,K)}
WK = ('W0','W1','W2')
FK = tuple(k for k in SHAPES if k not in WK)
ARMS = ('backprop','random_fixed','adaptive','warm_frozen')
LR, RATE_FB, WD_WB, WD_VU = .05, .8, .001, .01

def enc(w):
    w = w.flatten(); n = w.numel(); z = torch.fft.rfft(w, norm='ortho')
    mid = z[1:-1] if n % 2 == 0 else z[1:]
    out = [z[:1].real, torch.view_as_real(mid).flatten()*math.sqrt(2)]
    if n % 2 == 0: out.append(z[-1:].real)
    return torch.cat(out)

def dec(c):
    n = c.numel(); k = (n-1)//2
    mid = (c[1:1+2*k]/math.sqrt(2)).reshape(k,2).contiguous()
    out = [torch.complex(c[:1],torch.zeros_like(c[:1])),torch.view_as_complex(mid)]
    if n % 2 == 0: out.append(torch.complex(c[-1:],torch.zeros_like(c[-1:])))
    return torch.fft.irfft(torch.cat(out),n=n,norm='ortho')

def encode(dense): return {k: enc(v) for k,v in dense.items()}
def decode(state): return {k: dec(v).reshape(SHAPES[k]) for k,v in state.items()}
def clone(state): return {k:v.detach().clone() for k,v in state.items()}

def init(seed,dtype=torch.float32):
    gen=torch.Generator().manual_seed(seed)
    dense={}
    for k,shape in SHAPES.items():
        gain=math.sqrt(2) if k in ('W0','W1') else 1.
        dense[k]=torch.randn(*shape,generator=gen,dtype=dtype)*gain/math.sqrt(shape[1])
    return encode(dense)

def ports(w,x):
    h1=x@w['W0'].T; a1=h1.relu()
    h2=a1@w['W1'].T; a2=h2.relu()
    z=a2@w['W2'].T
    return [x,a1,a2,z], [h1>0,h2>0]

def forward(state,x): return ports(decode(state),x)[0][-1]

def loss_value(state,x,y): return F.binary_cross_entropy_with_logits(forward(state,x),y)

def dense_local(w,x,y,feedback=True):
    """Positive gradient-like drives; negative drives implement Eqs 1--4.

    Error includes 1/R; outer products include 1/batch exactly once.
    U2 is an identity readout connection, not a learned tensor.
    """
    a,gates=ports(w,x); n=len(x)
    s={3:(a[3].sigmoid()-y)/R}
    s[2]=s[3]@w['B2'].T; s[1]=s[2]@w['B1'].T
    g={'W2':s[3].T@a[2]/n}
    for j in (0,1):
        d=(s[j+1]@w['U'+str(j)].T)*gates[j]
        g['W'+str(j)]=d.T@a[j]/n
    for j in ((1,2) if feedback else ()):
        g['B'+str(j)]=RATE_FB*(a[j]@w['V'+str(j)].T).T@s[j+1]/n
        g['V'+str(j)]=RATE_FB*s[j].T@a[j]/n
        g['U'+str(j-1)]=RATE_FB*a[j].T@s[j]/n
    return g,a,gates,s

def exact_grads(state,x,y):
    with torch.enable_grad():
        q={k:v.detach().clone().requires_grad_(k in WK) for k,v in state.items()}
        loss=loss_value(q,x,y)
        grads=torch.autograd.grad(loss,[q[k] for k in WK])
    return {k:g.detach() for k,g in zip(WK,grads)}

def local_grads(state,x,y,feedback=True):
    with torch.no_grad():
        g,*_=dense_local(decode(state),x,y,feedback=feedback)
        return encode(g)

def apply(state,grads,keys):
    new=clone(state)
    for k in keys:
        wd=WD_WB if k.startswith(('W','B')) else WD_VU
        new[k]=state[k]-LR*(grads[k]+wd*state[k])
    return new

def update(state,x,y,arm):
    """No phase, test labels, source classification, or diagnostic gradients."""
    if arm=='backprop': return apply(state,exact_grads(state,x,y),WK)
    g=local_grads(state,x,y,feedback=arm=='adaptive')
    return apply(state,g,SHAPES if arm=='adaptive' else WK)

def dense_reference(w,x,y):
    """Independent per-example column-vector reference for the four rules."""
    out={k:torch.zeros_like(v) for k,v in w.items()}
    for xi,yi in zip(x,y):
        h1=w['W0']@xi; a1=h1.relu(); h2=w['W1']@a1; a2=h2.relu()
        z=w['W2']@a2; e3=(z.sigmoid()-yi)/R
        e2=w['B2']@e3; e1=w['B1']@e2
        out['W0']+=torch.outer((w['U0']@e1)*(h1>0),xi)
        out['W1']+=torch.outer((w['U1']@e2)*(h2>0),a1)
        out['W2']+=torch.outer(e3,a2)
        for j,aj,ej,en in [(1,a1,e1,e2),(2,a2,e2,e3)]:
            out['B'+str(j)]+=RATE_FB*torch.outer(w['V'+str(j)]@aj,en)
            out['V'+str(j)]+=RATE_FB*torch.outer(ej,aj)
            out['U'+str(j-1)]+=RATE_FB*torch.outer(aj,ej)
    return {k:v/len(x) for k,v in out.items()}

def world(seed):
    rng=np.random.default_rng(seed+61000000)
    angles=np.array([.2,1.2,2.2,2.9])+rng.uniform(-.15,.15,R)
    base=np.stack([np.cos(angles),np.sin(angles)],-1)
    novel=base@np.array([[0.,-1.],[1.,0.]])
    return {'slopes':np.stack([base if c<4 else novel for c in range(7)]).astype('float32'),
            'bias':rng.uniform(-.3,.3,(7,R)).astype('float32')}

def inputs(rng,contexts):
    contexts=np.asarray(contexts); x=np.zeros((len(contexts),D),dtype='float32')
    x[np.arange(len(contexts)),contexts]=1.
    x[:,7:9]=rng.uniform(-1,1,(len(x),2)); x[:,9:]=rng.uniform(-.2,.2,(len(x),2))
    return x

def labels(w,x,drift=False):
    ctx=x[:,:7].argmax(-1)
    y=((x[:,None,7:9]*w['slopes'][ctx]).sum(-1)+w['bias'][ctx]>0).astype('float32')
    if drift: y[ctx==0,:2]=1-y[ctx==0,:2]
    return y

def make_data(seed,warm_steps,phase_steps,query_n):
    w=world(seed); base_rng=np.random.default_rng(seed+62000000)
    bx=inputs(base_rng,np.repeat(np.arange(4),256)); by=labels(w,bx)
    warm_ix=base_rng.integers(len(bx),size=(warm_steps,64))
    rng=np.random.default_rng(seed+63000000); replay_rng=np.random.default_rng(seed+64000000)
    count=4*phase_steps
    cx=np.empty((count,16,D),dtype='float32'); cy=np.empty((count,16,R),dtype='float32')
    tx=np.empty((count,32,D),dtype='float32'); ty=np.empty((count,32,R),dtype='float32')
    init_idx=replay_rng.choice(len(bx),512,replace=False)
    memx=bx[init_idx].copy(); memy=by[init_idx].copy(); ids=-(init_idx+1)
    replay_ids=np.empty((count,16),dtype='int64'); stale_counts=np.zeros(count,dtype='int64')
    for t in range(count):
        phase=t//phase_steps
        if phase==0: ctx=rng.integers(0,4,16)
        elif phase==1: ctx=np.r_[rng.integers(4,6,12),rng.integers(0,4,4)]
        elif phase==2: ctx=np.r_[np.zeros(8,dtype=int),rng.integers(4,6,4),rng.integers(1,4,4)]
        else: ctx=rng.integers(0,6,16)
        x=inputs(rng,ctx); y=labels(w,x,phase>=2)
        ix=replay_rng.integers(len(memx),size=16)
        cx[t]=x;cy[t]=y;tx[t]=np.concatenate([x,memx[ix]]);ty[t]=np.concatenate([y,memy[ix]])
        replay_ids[t]=ids[ix]
        if phase>=2: stale_counts[t]=int((labels(w,memx[ix],True)!=memy[ix]).sum())
        memx=np.concatenate([memx[16:],x]); memy=np.concatenate([memy[16:],y])
        ids=np.concatenate([ids[16:],np.arange(16*t,16*t+16)])
    qrng=np.random.default_rng(seed+65000000)
    qx=inputs(qrng,np.repeat(np.arange(6),query_n)); ctx=qx[:,:7].argmax(-1)
    stable=np.repeat((ctx<4)[:,None],R,axis=1);stable[ctx==0,:2]=False
    novel=np.repeat((ctx>=4)[:,None],R,axis=1)
    drift=np.zeros_like(stable);drift[ctx==0,:2]=True
    data={'base_x':torch.tensor(bx),'base_y':torch.tensor(by),'warm_ix':torch.tensor(warm_ix),
          'current_x':torch.tensor(cx),'current_y':torch.tensor(cy),
          'train_x':torch.tensor(tx),'train_y':torch.tensor(ty),
          'query_x':torch.tensor(qx),'query_old_y':torch.tensor(labels(w,qx)),
          'query_new_y':torch.tensor(labels(w,qx,True)),
          'masks':{k:torch.tensor(v) for k,v in [('stable',stable),('novel',novel),('drift',drift)]},
          'replay_ids':torch.tensor(replay_ids),'stale_replayed_facts':torch.tensor(stale_counts),
          'world':{k:torch.tensor(v) for k,v in w.items()},'phase_steps':phase_steps,
          'query_n':query_n,'seed':seed}
    return data

@torch.no_grad()
def evaluate(state,data,drift):
    y=data['query_new_y' if drift else 'query_old_y']; z=forward(state,data['query_x'])
    ce=F.binary_cross_entropy_with_logits(z,y,reduction='none'); acc=((z>0)==y.bool()).float()
    out={}
    for k,m in data['masks'].items():
        out[k+'_ce']=float(ce[m].mean());out[k+'_acc']=float(acc[m].mean())
    out['balanced_acc']=sum(out[k+'_acc'] for k in data['masks'])/3
    return out

def flat(state,keys): return torch.cat([state[k].flatten() for k in keys])
def cosine(a,b):
    den=float(a.norm()*b.norm())
    return float(a@b)/den if den>1e-30 else None

def diagnostic(state,data,t,arm,initial_feedback):
    """Counterfactual probes on copies; never returned to the update path."""
    p=data['phase_steps']; n=data['query_n']
    ix=torch.cat([torch.arange(c*n,c*n+8) for c in range(6)])
    x=data['query_x'][ix]; y=data['query_new_y' if t>=2*p else 'query_old_y'][ix]
    exact=exact_grads(state,x,y)
    proposed=exact if arm=='backprop' else local_grads(state,x,y,feedback=False)
    before=float(loss_value(state,x,y)); candidate=apply(state,proposed,WK)
    after=float(loss_value(candidate,x,y))
    out={'t':t,'before_ce':before,'after_ce':after,'ce_change':after-before,
         'cosine':cosine(flat(exact,WK),flat(proposed,WK)),
         'norm_ratio':float(flat(proposed,WK).norm()/flat(exact,WK).norm().clamp_min(1e-30)),
         'layer_cosines':{k:cosine(exact[k],proposed[k]) for k in WK},
         'layer_norm_ratios':{k:float(proposed[k].norm()/exact[k].norm().clamp_min(1e-30)) for k in WK},
         'feedback_change_norm':float((flat(state,FK)-flat(initial_feedback,FK)).norm())}
    dense=decode(state); a,gates=ports(dense,x)
    out['interstream_mismatch']={str(j):float((dense['V'+str(j)]-dense['U'+str(j-1)].T).norm()) for j in (1,2)}
    out['interstream_cosines']={str(j):cosine(dense['V'+str(j)].flatten(),dense['U'+str(j-1)].T.flatten()) for j in (1,2)}
    out['pathway_relative_error']={}
    for j in (1,2):
        vv=dense['V'+str(j)]; ww=dense['W'+str(j)]
        # F_j(x)=V_j D_j W_j^T; B_j U_j^T; output U2=I.
        ff=torch.einsum('kh,bh,oh->bko',vv,gates[j-1].to(x.dtype),ww)
        bb=dense['B'+str(j)]@(dense['U'+str(j)].T if j==1 else torch.eye(R))
        out['pathway_relative_error'][str(j)]=float((ff-bb).norm()/ff.norm().clamp_min(1e-30))
    return out
