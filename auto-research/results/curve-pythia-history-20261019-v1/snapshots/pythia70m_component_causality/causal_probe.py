"""Native full-resolution curve interventions, with separate discovery/confirmation."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np
import torch
import torch.nn.functional as F
from scipy import fft
from safetensors import safe_open
from transformers import GPTNeoXForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
RAW=ROOT/'responses';RAW.mkdir(exist_ok=True)
P=json.loads((ROOT/'protocol.json').read_text())
DEFAULT=ROOT/'checkpoints'
PARAM='attention.query_key_value.weight';N=1536*512
CHECKPOINTS=DEFAULT
TEXT_DISC=[
 'The carpenter measured the shelf twice before cutting the board. She marked the position of each bracket and checked that the wall was level. The finished shelf held a row of books.',
 'During the night, snow covered the empty road. A driver cleared the windshield and waited for the engine to warm. By morning, sunlight had begun to melt the snow near the houses.',
 'A simple experiment needs a clear question and a way to measure the result. Keeping careful notes makes it easier to repeat the procedure. Unexpected measurements deserve a closer look.',
 'The shop sold bread, fruit, milk, and tea. A customer placed the groceries in a cloth bag and paid at the counter. Outside, a bicycle was leaning against the fence.'
]
TEXT_CONFIRM=[
 'A small stream flows through the meadow and into a pond. In spring the water rises after heavy rain. Frogs gather near the reeds, while insects hover above the surface.',
 'The mechanic opened the hood and inspected the belt. A loose connection explained the unusual noise. After making the repair, he started the engine again and listened carefully.',
 'The class compared two maps of the same town. One showed the roads and buildings, while the other showed changes in elevation. Together they helped explain where the river could flood.',
 'To prepare the soup, cut the carrots and onions into small pieces. Cook them gently before adding the stock. Stir the pot occasionally and taste it before adding more salt.',
 'A battery stores energy that can be used to power a circuit. The amount of current depends on the voltage and the resistance. A switch allows the circuit to be opened or closed.',
 'The traveler arrived early at the airport. She checked the departure board, found her gate, and sat beside a window. Workers were loading bags into the aircraft outside.',
 'The pianist repeated a difficult measure at a slow tempo. When both hands moved smoothly, he played a longer passage. Careful practice made the transition sound more natural.',
 'A librarian sorted the returned books into several groups. Some belonged on the shelves upstairs and others were reserved for readers. She placed a damaged volume on a separate cart.'
]

def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def read(name):return json.loads((OUT/name).read_text())
def tensor(step,layer):
    with safe_open(str(CHECKPOINTS/f'step{step}'/'model.safetensors'),framework='np') as f:
        return f.get_tensor(f'gpt_neox.layers.{layer}.{PARAM}')
def model(step):
    return GPTNeoXForCausalLM.from_pretrained(str(CHECKPOINTS/f'step{step}'),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
def param(m,layer):return m.gpt_neox.layers[layer].attention.query_key_value.weight

def haar_detail(x,level):
    a=np.asarray(x,np.float64)
    for j in range(1,level+1):
        d=(a[::2]-a[1::2])/np.sqrt(2);a=(a[::2]+a[1::2])/np.sqrt(2)
    return d
def packet(x,path):
    a=np.asarray(x,np.float64)
    for p in path:a=(a[::2]+(1 if p=='a' else -1)*a[1::2])/np.sqrt(2)
    return a
def ipacket(a,path):
    a=np.asarray(a,np.float64).copy()
    for p in path[::-1]:
        b=np.empty(len(a)*2);b[::2]=a/np.sqrt(2);b[1::2]=(1 if p=='a' else -1)*a/np.sqrt(2);a=b
    return a
def project(c,delta):
    x=delta[c['layer']].ravel();out=np.zeros_like(x)
    if c['basis']=='haar':
        j=c['level'];k=c['index'];size=2**j;lo=k*size;co=haar_detail(x,j)[k]
        out[lo:lo+size//2]=co/np.sqrt(size);out[lo+size//2:lo+size]=-co/np.sqrt(size)
    elif c['basis']=='fourier':
        z=fft.rfft(x,norm='ortho');z[:c['low']]=0;z[c['high']:]=0;out=fft.irfft(z,n=len(x),norm='ortho')
    elif c['basis']=='packet':
        lo,hi=c['start'],c['end'];out[lo:hi]=ipacket(packet(x[lo:hi],c['path']),c['path'])
    else:raise ValueError(c['basis'])
    return out.reshape(1536,512)

def prepare_inputs():
    rng=np.random.default_rng(P['input_seed']);pool=rng.permutation(np.arange(100,50000));off=0;sets={}
    for split,lengths in [('discovery',[32]*16),('confirmation',[16]*16+[32]*16+[48]*16)]:
        rows=[]
        for i,n in enumerate(lengths):
            seq=pool[off:off+n];off+=n;order=rng.permutation(n-1)
            rows.append(dict(id=i,n=n,prefix=seq.tolist(),order=order.tolist()))
        sets[split]=rows
    used0={x for r in sets['discovery'] for x in r['prefix']};used1={x for r in sets['confirmation'] for x in r['prefix']}
    assert not used0.intersection(used1)
    tok=AutoTokenizer.from_pretrained(str(CHECKPOINTS/'tokenizer'),local_files_only=True)
    payload={'sets':sets,'prose':{'discovery':TEXT_DISC,'confirmation':TEXT_CONFIRM},'prose_ids':{s:[tok(t).input_ids for t in texts] for s,texts in [('discovery',TEXT_DISC),('confirmation',TEXT_CONFIRM)]},
      'token_ids_disjoint':True,'description':'New original diagnostic prose, not a verified pretraining holdout.'}
    dump('probe_inputs.json',payload)

def catalog_fixed(delta):
    cats=[];triples=[]
    def add(c):cats.append(c);return c['id']
    for l in range(6):
        x=delta[l].ravel();z=fft.rfft(x,norm='ortho');k=int(np.argmax(abs(z)))
        add(dict(id=f'L{l}_fourier_atom',layer=l,basis='fourier',low=k,high=k+1,role='weight_selected_atom',label=f'L{l}: Fourier bin {k}'))
        for j in [6,9,15]:
            d=haar_detail(x,j);k=int(np.argmax(abs(d)))
            add(dict(id=f'L{l}_haar_j{j}',layer=l,basis='haar',level=j,index=k,role='weight_selected_atom',label=f'L{l}: Haar j{j}, k{k}'))
    add(dict(id='L3_all',layer=3,basis='packet',start=0,end=N,path='',role='global_root',label='L3: entire QKV interval update'))
    def packets(prefix,start,end,maxdepth,root_id=None,role='global_packet'):
        def rec(path):
            cid=root_id if path=='' and root_id else prefix+(path or 'root')
            if not (path=='' and root_id):add(dict(id=cid,layer=3,basis='packet',start=start,end=end,path=path,role=role,label=f'{prefix}: packet {path or "root"}'))
            if len(path)<maxdepth:
                a=rec(path+'a');b=rec(path+'d');triples.append(dict(parent=cid,left=a,right=b,kind='packet'))
            return cid
        rec('')
    packets('G_',0,N,2,'L3_all')
    def frec(lo,hi,depth,path):
        cid='L3_all' if not path else 'F_'+path
        if path:add(dict(id=cid,layer=3,basis='fourier',low=lo,high=hi,role='global_fourier',label=f'L3: Fourier bins {lo}–{hi-1}'))
        if depth<2:
            mid=(lo+hi)//2;a=frec(lo,mid,depth+1,path+'a');b=frec(mid,hi,depth+1,path+'b');triples.append(dict(parent=cid,left=a,right=b,kind='fourier'))
        return cid
    frec(0,N//2+1,0,'')
    packets('H6_',589824,655360,2,role='local_packet')
    for h in [5,7]:
        lo=h*3*64*512
        add(dict(id=f'H{h}_root',layer=3,basis='packet',start=lo,end=lo+65536,path='',role='neighbor_control',label=f'L3 head {h}: native Q/K interval'))
    for name,lo,hi in [('H6_query',589824,622592),('H6_key',622592,655360)]:
        add(dict(id=name,layer=3,basis='packet',start=lo,end=hi,path='',role='projection_control',label=name.replace('_',' ')))
    triples.append(dict(parent='H6_root',left='H6_query',right='H6_key',kind='native_spatial_split'))
    return cats,triples

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

@torch.inference_mode()
def save_case(m,split,phase,cid,reference=None,meta=None):
    score,h,teachers=evaluate(m,split,reference)
    row=dict(id=cid,phase=phase,split=split,**(meta or {}),**score)
    np.savez_compressed(RAW/f'{split}_{phase}_{cid}.npz',**h)
    return row,teachers

def metric(row):return row['repeat']['target_nll']

@torch.inference_mode()
def discover():
    begin=time.time();prepare_inputs()
    original=[tensor(1000,l) for l in range(6)];old=[tensor(512,l) for l in range(6)]
    delta=[b.astype(np.float64)-a for a,b in zip(old,original)]
    np.savez_compressed(ROOT/'native_qkv_curves.npz',**{f'step{s}_layer{l}':v for s,vs in [(512,old),(1000,original)] for l,v in enumerate(vs)})
    cats,triples=catalog_fixed(delta);m=model(1000)
    baseline,teachers=save_case(m,'discovery','rollback','baseline');rows=[baseline];scores={}
    print('discovery baseline',json.dumps({k:baseline[k]['accuracy'] for k in ['repeat','scrambled']}),flush=True)
    def run(c):
        t=time.time();p=param(m,c['layer']);v=project(c,delta)
        p.copy_(torch.from_numpy((original[c['layer']].astype(np.float64)-v).astype(np.float32)))
        row,_=save_case(m,'discovery','rollback',c['id'],teachers,dict(update_norm=float(np.linalg.norm(v)),max_parameter_change=float(abs(v).max())))
        p.copy_(torch.from_numpy(original[c['layer']]));assert torch.equal(p,torch.from_numpy(original[c['layer']]))
        rows.append(row);scores[c['id']]=metric(row)-metric(baseline)
        print(c['id'],'repeat NLL +',round(scores[c['id']],6),'acc',round(row['repeat']['accuracy'],4),'seconds',round(time.time()-t,2),flush=True)
        dump('discovery.json',rows)
    for c in cats:run(c)
    depth2=[c for c in cats if c['role']=='local_packet' and len(c['path'])==2]
    current=max(depth2,key=lambda c:scores[c['id']]);descent=[]
    for depth in range(3,7):
        if scores[current['id']]<.01:break
        pair=[]
        for suffix in ['a','d']:
            path=current['path']+suffix;c=dict(id='H6_'+path,layer=3,basis='packet',start=589824,end=655360,path=path,role='adaptive_local_packet',label=f'Head-6 Q/K: packet {path}')
            cats.append(c);run(c);pair.append(c)
        triples.append(dict(parent=current['id'],left=pair[0]['id'],right=pair[1]['id'],kind='packet'))
        descent.append(dict(parent=current['id'],children=[c['id'] for c in pair]))
        current=max(pair,key=lambda c:scores[c['id']])
    eligible=[c for c in cats if c['role'] in ['weight_selected_atom','global_packet','global_fourier','local_packet','adaptive_local_packet']]
    top=sorted(eligible,key=lambda c:scores[c['id']],reverse=True)[:6]
    selected={c['id'] for c in top}|{'L3_all','H6_root','H6_query','H6_key','H5_root','H7_root','L3_fourier_atom','L3_haar_j6'}
    selected|={c['id'] for c in cats if c['role'] in ['local_packet','adaptive_local_packet']}
    # Include complete sibling triples for selected coarse global nodes, enabling interaction checks.
    for tr in triples:
        if tr['parent'] in selected and tr['parent']!='L3_all':selected|={tr['left'],tr['right']}
    selection=dict(candidate_ids=[c['id'] for c in cats if c['id'] in selected],top6=[c['id'] for c in top],local_descent=descent,strongest_local_leaf=current['id'],
      rule=P['selection'],confirmation_consulted=False,later_context_ids=['H6_root',current['id'],'L3_fourier_atom','L3_haar_j6'])
    dump('candidate_catalog.json',cats);dump('sibling_triples.json',triples);dump('selection_locked.json',selection)
    # Assert exact parameter restoration and projection additivity before confirmation.
    numerical=[];byid={c['id']:c for c in cats}
    for tr in triples:
        a=project(byid[tr['parent']],delta);b=project(byid[tr['left']],delta);c=project(byid[tr['right']],delta)
        err=float(np.max(abs(a-b-c)));assert err<1e-12
        numerical.append(dict(**tr,max_abs_parameter_nonadditivity=err))
    dump('parameter_additivity.json',numerical)
    restored,_,_=evaluate(m,'discovery',teachers)
    err=max(abs(restored[k]['target_nll']-baseline[k]['target_nll']) for k in ['repeat','scrambled']);assert err==0
    dump('discovery_environment.json',dict(seconds=time.time()-begin,candidates=len(cats),selected=len(selection['candidate_ids']),restore_nll_difference=err,training_updates=0))
    print('DISCOVERY COMPLETE',json.dumps(selection),flush=True)

@torch.inference_mode()
def confirmation():
    begin=time.time();selection=read('selection_locked.json');selection_hash=hashlib.sha256((OUT/'selection_locked.json').read_bytes()).hexdigest()
    cats={c['id']:c for c in read('candidate_catalog.json')};old=[tensor(512,l) for l in range(6)];new=[tensor(1000,l) for l in range(6)]
    delta=[b.astype(np.float64)-a for a,b in zip(old,new)];rows=[];checks=[]
    for phase,step,base,sign in [('rollback',1000,new,-1),('transplant',512,old,1)]:
        m=model(step);baseline,teachers=save_case(m,'confirmation',phase,'baseline');rows.append(baseline)
        w=m.embed_out.weight.double();mean=w.mean(0);gram=w.T@w/w.shape[0]-torch.outer(mean,mean)
        np.save(OUT/f'logit_gram_step{step}.npy',gram.numpy());del w,gram
        print('confirmation',phase,'baseline',baseline['repeat']['accuracy'],baseline['scrambled']['accuracy'],flush=True)
        for cid in selection['candidate_ids']:
            c=cats[cid];v=project(c,delta);p=param(m,c['layer']);p.copy_(torch.from_numpy((base[c['layer']].astype(np.float64)+sign*v).astype(np.float32)))
            row,_=save_case(m,'confirmation',phase,cid,teachers,dict(update_norm=float(np.linalg.norm(v))))
            p.copy_(torch.from_numpy(base[c['layer']]));assert torch.equal(p,torch.from_numpy(base[c['layer']]))
            rows.append(row);dump('confirmation.json',rows)
            print(phase,cid,'repeat',round(row['repeat']['accuracy'],4),'scrambled',round(row['scrambled']['accuracy'],4),'prose Δ',round(row['prose']['nll']-baseline['prose']['nll'],6),flush=True)
        if phase=='rollback':
            def zero_head(module,args):
                x=args[0].clone();x[...,6*64:7*64]=0;return (x,)+args[1:]
            hook=m.gpt_neox.layers[3].attention.dense.register_forward_pre_hook(zero_head)
            row,_=save_case(m,'confirmation',phase,'head6_zero',teachers);hook.remove();rows.append(row);dump('confirmation.json',rows)
        restored,_,_=evaluate(m,'confirmation',teachers)
        error=max(abs(restored[k]['target_nll']-baseline[k]['target_nll']) for k in ['repeat','scrambled'])
        assert error==0
        checks.append(dict(step=step,phase=phase,restore_nll_difference=error))
        del m,teachers;gc.collect()
    m=model(8000);late=[tensor(8000,l) for l in range(6)];ld=[b.astype(np.float64)-a for a,b in zip(old,late)]
    baseline,teachers=save_case(m,'confirmation','later','baseline');later=[baseline]
    for cid in selection['later_context_ids']:
        c=cats[cid];v=project(c,ld);p=param(m,c['layer']);p.copy_(torch.from_numpy((late[c['layer']].astype(np.float64)-v).astype(np.float32)))
        row,_=save_case(m,'confirmation','later',cid,teachers,dict(update_norm=float(np.linalg.norm(v))))
        p.copy_(torch.from_numpy(late[c['layer']]));later.append(row);dump('later_context.json',later)
        print('later',cid,row['repeat']['accuracy'],row['scrambled']['accuracy'],flush=True)
    assert selection_hash==hashlib.sha256((OUT/'selection_locked.json').read_bytes()).hexdigest()
    dump('confirmation_environment.json',dict(seconds=time.time()-begin,variants=len(rows)+len(later),selection_sha256=selection_hash,selection_changed=False,restore_checks=checks,
      training_updates=0,torch=torch.__version__,threads=torch.get_num_threads(),dtype='float32 inference; float64 transforms'))
    print('CONFIRMATION COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['discover','confirm']);ap.add_argument('--checkpoints',type=Path,default=DEFAULT);a=ap.parse_args()
    CHECKPOINTS=a.checkpoints;torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    discover() if a.phase=='discover' else confirmation()
