"""Scoring and native packet transforms, retained from the verified previous phase."""
import os,json
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import GPTNeoXForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';RAW=ROOT/'responses'
CHECKPOINTS=ROOT/'checkpoints'
PARAM='attention.query_key_value.weight'
def read(name):return json.loads((OUT/name).read_text())
def tensor(step,layer):
    with safe_open(str(CHECKPOINTS/f'step{step}'/'model.safetensors'),framework='np') as f:
        return f.get_tensor(f'gpt_neox.layers.{layer}.{PARAM}')

def model(step):
    return GPTNeoXForCausalLM.from_pretrained(str(CHECKPOINTS/f'step{step}'),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()

def param(m,layer):return m.gpt_neox.layers[layer].attention.query_key_value.weight

def packet(x,path):
    a=np.asarray(x,np.float64)
    for p in path:a=(a[::2]+(1 if p=='a' else -1)*a[1::2])/np.sqrt(2)
    return a

def ipacket(a,path):
    a=np.asarray(a,np.float64).copy()
    for p in path[::-1]:
        b=np.empty(len(a)*2);b[::2]=a/np.sqrt(2);b[1::2]=(1 if p=='a' else -1)*a/np.sqrt(2);a=b
    return a

@torch.inference_mode()
def evaluate(m,split,reference=None):
    data=read('probe_inputs.json');seqs=data['sets'][split];out={};saved={}
    for kind in ['repeat','scrambled']:
        rows=[];hidden=[];orders=[]
        for n in sorted({r['n'] for r in seqs}):
            group=[r for r in seqs if r['n']==n]
            for start in range(0,len(group),4):
                batch=group[start:start+4];ids=[];successors=[];perms=[]
                for r in batch:
                    seq=np.array(r['prefix']);order=np.arange(n-1) if kind=='repeat' else np.array(r['order'])
                    ids.append(np.r_[0,seq,seq[order]]);successors.append(seq[order+1]);perms.append(order)
                ids=torch.tensor(np.array(ids));q=torch.arange(n+1,2*n);fp=torch.arange(1,n)
                net=m.gpt_neox(ids,use_cache=False,output_attentions=True)
                h=net.last_hidden_state
                # This is the same backbone + output projection used by GPTNeoXForCausalLM.forward.
                logits=m.embed_out(h[:,torch.cat([fp,q])]);lp=F.log_softmax(logits,dim=-1)
                for bi,r in enumerate(batch):
                    targets=torch.tensor(successors[bi]);first=ids[bi,fp+1]
                    nll=-lp[bi,n-1:,targets].diagonal();fnll=-lp[bi,:n-1,first].diagonal()
                    attn=net.attentions[3][bi,6];keys=torch.tensor(2+perms[bi]);dup=keys-1
                    rows.append(dict(id=r['id'],n=n,target_nll=float(nll.mean()),accuracy=float((logits[bi,n-1:].argmax(-1)==targets).float().mean()),
                      first_nll=float(fnll.mean()),advantage=float((fnll-nll).mean()),content_attention=float(attn[q,keys].mean()),duplicate_attention=float(attn[q,dup].mean()),
                      position_nll=nll.tolist(),position_correct=(logits[bi,n-1:].argmax(-1)==targets).tolist()))
                    hidden.append(h[bi,q[-1]].numpy().copy());orders.append(r['id'])
                del logits,lp,net
        ii=np.argsort(orders);rows=[rows[i] for i in ii];saved[kind+'_hidden']=np.stack(hidden)[ii]
        out[kind]={'sequences':rows,**{k:float(np.mean([r[k] for r in rows])) for k in ['target_nll','accuracy','first_nll','advantage','content_attention','duplicate_attention']}}
    prows=[];teachers=[]
    for j,ids0 in enumerate(data['prose_ids'][split]):
        ids=torch.tensor([ids0]);logits=m(ids,use_cache=False).logits[0,:-1];lp=F.log_softmax(logits,dim=-1)
        losses=-lp[torch.arange(len(ids0)-1),ids[0,1:]]
        row=dict(id=j,tokens=len(losses),nll=float(losses.mean()))
        if reference is not None:
            ref=reference[j];kl=torch.sum(ref.exp().double()*(ref-lp).double(),dim=-1).mean()
            row.update(kl_from_baseline=float(kl),argmax_changed=float((lp.argmax(-1)!=ref.argmax(-1)).float().mean()))
        else:row.update(kl_from_baseline=0.,argmax_changed=0.);teachers.append(lp.clone())
        prows.append(row)
    total=sum(r['tokens'] for r in prows)
    out['prose']={'passages':prows,'tokens':total,**{k:sum(r['tokens']*r[k] for r in prows)/total for k in ['nll','kl_from_baseline','argmax_changed']}}
    return out,saved,teachers
