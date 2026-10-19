"""Inference-only probes; no optimizer and no model training."""
import gc, hashlib, time, platform, numpy as np, torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from common import *

torch.set_num_threads(3)
torch.set_num_interop_threads(1)

TEXTS=[
 'The network engineer checked the switch configuration before the maintenance window. A new cable connected the building to the central office. After saving a copy of the settings, the engineer tested each connection and recorded the results in a notebook.',
 'Rain began during the afternoon, and the paths through the park became muddy. Maya closed the kitchen window and put a kettle on the stove. She planned to read for an hour before walking to the library to return her books.',
 'A telescope gathers light from distant objects. A larger mirror can collect more light, but the quality of the image also depends on the atmosphere and the instrument. Astronomers compare several observations before deciding whether a faint signal is real.',
 'The recipe called for flour, water, salt, and yeast. After mixing the ingredients, Sam covered the bowl and left the dough in a warm place. It gradually expanded. He shaped it into a loaf and heated the oven.',
 'In a controlled experiment, researchers change one factor while holding the others as constant as possible. Repeating the measurement helps reveal ordinary variation. A surprising result is a reason to inspect the method and collect more evidence.',
 'The train stopped at a small station near the river. Passengers stepped onto the platform carrying bags and folded umbrellas. Across the street, a cafe had just opened. The smell of fresh coffee drifted through its doorway.',
 'A computer program can store a collection of numbers in an array. Each position has an index. To find the largest number, the program compares the current candidate with each remaining entry and keeps the larger value.',
 'At the meeting, the team reviewed the previous week of work. One task was complete, two needed more testing, and a fourth depended on a delivery. They wrote down the decisions so everyone could check the plan later.',
 'The garden was quiet in the morning. Small birds searched the ground beneath the bushes while sunlight reached the tops of the trees. A neighbor carried a watering can along the fence and stopped to inspect the tomatoes.',
 'A fraction represents a part of a whole. When two fractions have the same denominator, their numerators can be added directly. If the denominators differ, an equivalent pair with a common denominator makes the addition possible.',
 'The musician practiced a short passage slowly, listening for an even rhythm. When the notes felt comfortable, she increased the tempo a little. She then played the surrounding measures to check that the passage fit naturally into the piece.',
 'The museum displayed tools used by people who lived in the valley long ago. Labels described how each object was made and where it had been found. Visitors could compare the worn originals with drawings of their likely appearance when new.'
]

def load_model(step):
    return GPTNeoXForCausalLM.from_pretrained(
      str(ROOT/'checkpoints'/f'step{step}'),local_files_only=True,
      torch_dtype=torch.float32,attn_implementation='eager').eval()

def text_inputs():
    tok=AutoTokenizer.from_pretrained(str(ROOT/'checkpoints'/'tokenizer'),local_files_only=True)
    return [tok(t,return_tensors='pt').input_ids for t in TEXTS]

@torch.inference_mode()
def text_score(model, inputs, teachers=None, keep_teacher=False):
    rows=[]; saved=[]
    for i,ids in enumerate(inputs):
        logits=model(ids,use_cache=False).logits[:,:-1,:].float()
        logp=F.log_softmax(logits,dim=-1)
        loss=F.nll_loss(logp.reshape(-1,logp.size(-1)),ids[:,1:].reshape(-1),reduction='none')
        row={'passage':i,'tokens':loss.numel(),'nll':float(loss.mean())}
        if teachers is not None:
            target=teachers[i]
            row['kl_to_target']=float(torch.sum(target.exp()*(target-logp),dim=-1).mean())
        if keep_teacher: saved.append(logp.clone())
        rows.append(row)
    n=sum(r['tokens'] for r in rows)
    out={'nll':sum(r['tokens']*r['nll'] for r in rows)/n,'tokens':n,'passages':rows}
    if teachers is not None: out['kl_to_target']=sum(r['tokens']*r['kl_to_target'] for r in rows)/n
    return (out,saved) if keep_teacher else out

def make_sequences():
    rng=np.random.default_rng(SEED); n=32
    seq=np.stack([rng.choice(np.arange(100,50000),n,replace=False) for _ in range(32)])
    permutations=np.stack([rng.permutation(n) for _ in range(32)])
    repeated=np.concatenate([np.zeros((32,1),dtype=np.int64),seq,seq],axis=1)
    shuffled=np.concatenate([np.zeros((32,1),dtype=np.int64),seq,np.take_along_axis(seq,permutations,axis=1)],axis=1)
    np.savez(OUT/'behavior_inputs.npz',repeated=repeated,shuffled=shuffled,permutations=permutations)
    return repeated,shuffled,permutations

@torch.inference_mode()
def induction_score(model,repeated,shuffled,permutations):
    results={}; n=32
    for name,data in [('repeated',repeated),('shuffled',shuffled)]:
        head_content=[]; head_lag=[]; head_duplicate=[]; seqrows=[]
        for start in range(0,len(data),4):
            ids=torch.tensor(data[start:start+4]); out=model(ids,use_cache=False,output_attentions=True)
            logp=F.log_softmax(out.logits.float(),dim=-1)
            for b in range(len(ids)):
                firstpos=torch.arange(1,n)
                secondpos=torch.arange(n+1,2*n)
                first=-logp[b,firstpos,ids[b,firstpos+1]]
                second=-logp[b,secondpos,ids[b,secondpos+1]]
                seqrows.append({'sequence':start+b,'first_nll':float(first.mean()),'second_nll':float(second.mean()),
                  'copy_advantage_nats':float((first-second).mean()),
                  'second_top1':float((out.logits[b,secondpos].argmax(-1)==ids[b,secondpos+1]).float().mean())})
                orig=np.arange(n) if name=='repeated' else permutations[start+b]
                positions=np.flatnonzero(orig<n-1)
                qs=torch.tensor(n+1+positions)
                ks=torch.tensor(2+orig[positions])
                dc=torch.tensor(1+orig[positions])
                lag=qs-(n-1)
                head_content.append(np.stack([a[b,:,qs,ks].mean(-1).numpy() for a in out.attentions]))
                head_lag.append(np.stack([a[b,:,qs,lag].mean(-1).numpy() for a in out.attentions]))
                head_duplicate.append(np.stack([a[b,:,qs,dc].mean(-1).numpy() for a in out.attentions]))
            del out,logp
        content=np.mean(head_content,axis=0)
        flat_idx=int(np.argmax(content))
        results[name]={'sequences':seqrows,'copy_advantage_nats':float(np.mean([r['copy_advantage_nats'] for r in seqrows])),
          'first_nll':float(np.mean([r['first_nll'] for r in seqrows])),
          'second_nll':float(np.mean([r['second_nll'] for r in seqrows])),
          'second_top1':float(np.mean([r['second_top1'] for r in seqrows])),
          'content_following_attention':content.tolist(),
          'fixed_lag_attention':np.mean(head_lag,axis=0).tolist(),
          'duplicate_token_attention':np.mean(head_duplicate,axis=0).tolist(),
          'max_content_attention':float(content.max()),'max_layer':flat_idx//8,'max_head':flat_idx%8}
    return results

@torch.inference_mode()
def permutation_check(model,orders,ids):
    base=model(ids,use_cache=False).logits.clone()
    checks=[]
    for name in ['random0','pca_frozen']:
        saved=[]
        for l,block in enumerate(model.gpt_neox.layers):
            p=torch.tensor(orders[(l,name)])
            win=block.mlp.dense_h_to_4h; wout=block.mlp.dense_4h_to_h
            saved.append((win.weight.clone(),win.bias.clone(),wout.weight.clone()))
            win.weight.copy_(win.weight[p]); win.bias.copy_(win.bias[p]); wout.weight.copy_(wout.weight[:,p])
        cand=model(ids,use_cache=False).logits
        diff=cand-base
        checks.append({'ordering':name,'max_abs_logit_difference':float(diff.abs().max()),
          'relative_rms_logit_difference':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(base)),
          'argmax_agreement':float((cand.argmax(-1)==base.argmax(-1)).float().mean()),'positions':ids.numel()})
        for block,(wi,bi,wo) in zip(model.gpt_neox.layers,saved):
            block.mlp.dense_h_to_4h.weight.copy_(wi)
            block.mlp.dense_h_to_4h.bias.copy_(bi)
            block.mlp.dense_4h_to_h.weight.copy_(wo)
            assert torch.equal(block.mlp.dense_h_to_4h.weight,wi)
        assert checks[-1]['relative_rms_logit_difference']<1e-5
    return checks

@torch.inference_mode()
def sparse_interventions(inputs):
    orders=get_orderings(); model=load_model(8000)
    source=state(4000)
    initial={}; target={}; delta={}; params={}
    for l,block in enumerate(model.gpt_neox.layers):
        for fam,attr in [('mlp_in','dense_h_to_4h'),('mlp_out','dense_4h_to_h')]:
            k=(l,fam); param=getattr(block.mlp,attr).weight
            params[k]=param
            target[k]=param.numpy().copy()
            initial[k]=source[f'gpt_neox.layers.{l}.mlp.{attr}.weight'].copy()
            delta[k]=target[k]-initial[k]
    del source
    full,teachers=text_score(model,inputs,keep_teacher=True)
    repeated,_,_=make_sequences()
    checks=permutation_check(model,orders,torch.tensor(repeated[:4]))
    dump('permutation_checks.json',checks)
    def install(updates):
        for k,param in params.items(): param.copy_(torch.from_numpy(initial[k]+updates[k]))
    install({k:np.zeros_like(v) for k,v in delta.items()})
    reverted=text_score(model,inputs,teachers)
    rows=[{'variant':'full_target','fraction':1.,**full,'kl_to_target':0.},
          {'variant':'reverted_mlp_weights','fraction':0.,**reverted}]
    variants=[('spatial_top','native','spatial','top'),('spatial_random','native','spatial','random'),
      ('fourier_native_top','native','fourier','top'),('fourier_random0_top','random0','fourier','top'),
      ('fourier_random1_top','random1','fourier','top'),('fourier_random2_top','random2','fourier','top'),
      ('fourier_pca_top','pca_frozen','fourier','top'),('fourier_native_low','native','fourier','low'),
      ('fourier_pca_low','pca_frozen','fourier','low')]
    denom=sum(energy(x) for x in delta.values())
    for fraction in [.1,.25]:
        for label,order,basis,mode in variants:
            t=time.time(); upd={}; count=0
            for (l,fam),d in delta.items():
                axis=axis_for(fam); p=orders[(l,order)]
                y=ordered(d,fam,p)
                c=real_fft(y,axis) if basis=='fourier' else y
                masked,kept=mask_coefficients(c,fraction,mode,np.random.default_rng(SEED+l+(0 if fam=='mlp_in' else 100)),axis)
                rec=real_ifft(masked,axis) if basis=='fourier' else masked
                upd[(l,fam)]=unordered(rec,fam,p); count+=kept
            install(upd); score=text_score(model,inputs,teachers)
            row={'variant':label,'fraction':fraction,'real_coordinates_kept':count,
              'energy_retained':sum(energy(x) for x in upd.values())/denom,
              'kl_recovery_fraction':1-score['kl_to_target']/reverted['kl_to_target'],
              'nll_recovery_fraction':(reverted['nll']-score['nll'])/(reverted['nll']-full['nll']),**score}
            rows.append(row); dump('sparse_interventions.json',rows)
            print('sparse',label,fraction,'KL',score['kl_to_target'],'NLL',score['nll'],f'{time.time()-t:.1f}s',flush=True)
    install(delta)
    restored=text_score(model,inputs,teachers)
    dump('endpoint_check.json',{'full_nll':full['nll'],'restored_nll':restored['nll'],
      'restored_kl':restored['kl_to_target'],'mlp_weight_parameters':sum(x.size for x in delta.values())})
    assert abs(restored['nll']-full['nll'])<1e-5
    assert abs(restored['kl_to_target'])<1e-6
    return rows

def main():
    begin=time.time(); inputs=text_inputs()
    dump('prose_probe.json',{'texts':TEXTS,'tokens_per_passage':[int(t.numel()-1) for t in inputs],
      'description':'Original short prose written for this diagnostic. Not a standardized benchmark or an independently verified pretraining holdout.'})
    repeated,shuffled,perms=make_sequences(); rows=[]
    for step in STEPS:
        t=time.time(); model=load_model(step)
        probes=induction_score(model,repeated,shuffled,perms)
        prose=text_score(model,inputs)
        row={'step':step,'prose':prose,**probes}; rows.append(row)
        dump('behavior.json',rows)
        print('behavior',step,'copy advantage',probes['repeated']['copy_advantage_nats'],
          'max induction',probes['repeated']['max_content_attention'],'NLL',prose['nll'],f'{time.time()-t:.1f}s',flush=True)
        del model; gc.collect()
    sparse_interventions(inputs)
    dump('behavior_environment.json',{'seconds':time.time()-begin,'torch':torch.__version__,'threads':torch.get_num_threads(),
      'optimizer_steps':0,'precision':'float32 inference, eager attention','total_prose_targets':sum(t.numel()-1 for t in inputs)})
    print('BEHAVIOR COMPLETE',flush=True)

if __name__=='__main__': main()
