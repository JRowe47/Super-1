"""Frozen task definitions. Task identity is an explicit caller choice, not a token heuristic."""
import hashlib,itertools,json,random
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
PARENT=H.parent/'adaptive-depth-shakespeare-20261018-v1'

def record(prompt,answer,context):
    seq=list((prompt+answer).encode('ascii')); at=len(prompt)-1
    assert len(seq)<=context+1
    x=seq[:-1]+[0]*(context-len(seq)+1); y=seq[1:]+[0]*(context-len(seq)+1)
    mask=[False]*context; mask[at]=True
    return {'x':x,'y':y,'mask':mask,'prompt':prompt,'answer':answer}

def generate(c):
    rng=random.Random(c['data_seed']); sets={}
    cases=list(itertools.product(range(c['modulus']),repeat=c['arithmetic_operands']))
    rng.shuffle(cases)
    arithmetic=[record('+'.join(map(str,v))+f" mod {c['modulus']}=",str(sum(v)%c['modulus']),c['context']) for v in cases]
    total=c['synthetic_train_examples']+c['synthetic_tune_examples']+c['selection_examples']+c['confirmation_examples']
    seen=set(); retrieval=[]
    while len(retrieval)<total:
        keys=rng.sample(c['retrieval_keys'],c['retrieval_pairs']); values=[rng.randrange(c['modulus']) for _ in keys]
        query=rng.choice(keys); prompt=';'.join(f'{k}={v}' for k,v in zip(keys,values))+f';query={query};answer='
        if prompt in seen:continue
        seen.add(prompt); retrieval.append(record(prompt,str(values[keys.index(query)]),c['context']))
    sizes=[c['synthetic_train_examples'],c['synthetic_tune_examples'],c['selection_examples'],c['confirmation_examples']]
    for name,rows in [('arithmetic',arithmetic),('retrieval',retrieval)]:
        sets[name]={}; cursor=0
        for split,n in zip(['train','tune','selection','confirmation'],sizes):
            sets[name][split]=rows[cursor:cursor+n];cursor+=n
        assert len(set(r['prompt'] for parts in sets[name].values() for r in parts))==sum(sizes)
    return sets

def load_data(c):
    saved=json.loads((H/'DATA.json').read_text()); result={}
    for task,parts in saved.items():
        result[task]={split:tuple(torch.tensor([r[k] for r in rows],dtype=torch.bool if k=='mask' else torch.long) for k in ['x','y','mask']) for split,rows in parts.items()}
    raw=(PARENT/'corpus.txt').read_bytes()
    result['text']={s:torch.tensor(list(raw[a:b]),dtype=torch.long) for s,(a,b) in c['text_regions'].items()}
    return result

def batch(data,c,task,split,generator=None,indices=None):
    if task=='text':
        d=data[task][split]
        if indices is None:indices=torch.randint(len(d)-c['context'],(c['batch'],),generator=generator)
        offsets=indices[:,None]+torch.arange(c['context'])[None,:]
        x,y=d[offsets].contiguous(),d[offsets+1].contiguous()
        return x,y,torch.ones_like(x,dtype=torch.bool)
    parts=data[task][split]
    if indices is None:indices=torch.randint(len(parts[0]),(c['batch'],),generator=generator)
    return tuple(p[indices].contiguous() for p in parts)

def evaluation(data,c,task,split):
    n=c[split+'_examples']; idx=torch.arange(n)
    if task=='text':idx=idx*(c['context']+1)
    return batch(data,c,task,split,indices=idx)
