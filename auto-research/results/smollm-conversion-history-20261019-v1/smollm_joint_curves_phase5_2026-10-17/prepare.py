import gc,hashlib,json,re,time,urllib.request
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from common import ROOT,OUT,SEED,dump,setup,model_dir,load_model,data,hashes
from curve_model import restore
from joint_model import make_banks,set_model,basis_analysis,TIMES,sampled_delta_rms,reset_delta


def select_data():
    if (ROOT/'data/token_windows.npz').exists():return
    p3=json.loads((ROOT/'reference/phase3_dataset_manifest.json').read_text())
    p2=json.loads((ROOT/'reference/phase2_dataset_manifest.json').read_text())
    with np.load(ROOT.parent/'smollm_depth_curve_phase3/data/token_windows.npz') as a:
        arrays={'train':a['train'].copy(),'validation':a['validation'].copy(),'legacy_test':a['test'][:64].copy()}
    excluded={r['document_index'] for p in [p2,p3] for r in p['splits']['train']['selected_windows']}
    path=ROOT/'cache/train.parquet'
    if not path.exists():
        url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{p3["revision"]}/wikitext-2-raw-v1/train/0000.parquet'
        with urllib.request.urlopen(url,timeout=90) as response:path.write_bytes(response.read())
    sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==p3['files']['train.parquet']['sha256']
    docs=[];parts=[];title=None
    for line in pq.read_table(path).column('text').to_pylist():
        match=re.match(r'^\s*= ([^=].*?) =\s*$',line)
        if match:
            if parts:docs.append((title or '[untitled]','\n'.join(parts)))
            title=match.group(1).strip();parts=[line]
        else:parts.append(line)
    if parts:docs.append((title or '[untitled]','\n'.join(parts)))
    tok=AutoTokenizer.from_pretrained(model_dir(),local_files_only=True);tok.model_max_length=10**9
    eligible={}
    for i,(title,text) in enumerate(docs):
        if i in excluded:continue
        ids=tok(text,add_special_tokens=False)['input_ids']
        starts=list(range(0,len(ids)-256,257))
        if len(starts)>=2:eligible[i]=(title,ids,starts)
    assert len(eligible)>=32,(len(eligible),len(excluded))
    rng=np.random.default_rng(SEED+1);chosen=rng.choice(sorted(eligible),32,replace=False)
    audit=[];origins=[]
    for doc in chosen:
        title,ids,starts=eligible[int(doc)]
        for offset in rng.choice(starts,2,replace=False):
            audit.append(ids[int(offset):int(offset)+257]);origins.append({'document_index':int(doc),'title':title,'token_offset':int(offset),'official_split':'train'})
    arrays['audit_long']=np.asarray(audit,dtype=np.int32)
    arrays['audit']=arrays['audit_long'][:,128:].copy()
    np.savez_compressed(ROOT/'data/token_windows.npz',**arrays)
    manifest={'repository':p3['repository'],'config':p3['config'],'revision':p3['revision'],
        'train_parquet_sha256':sha,'train_windows':1024,'validation_windows':32,
        'audit_windows':64,'audit_documents':32,'audit_sampling_seed':SEED+1,
        'audit_source':'Previously unused documents from the official TRAIN partition, reserved as this experiment\'s fresh document-held-out audit. This is not the official benchmark test split.',
        'audit_exclusions':'Union of every official-train article sampled in phases 2 and 3 (phase 4 was a subset). No audit document used for optimization or checkpoint selection.',
        'audit_pretraining_limit':'No claim that the reference model never saw these articles during its original pretraining.',
        'excluded_train_document_indices':sorted(excluded),'eligible_audit_documents':len(eligible),
        'audit_windows_manifest':origins,'train_windows_manifest':p3['splits']['train']['selected_windows'],
        'validation_windows_manifest':p3['splits']['validation']['selected_windows'],
        'legacy_test_windows_manifest':p3['splits']['test']['selected_windows'][:64],
        'long_context':'Same 128 target tokens as audit; the 256-context run supplies an additional 128 preceding tokens. Score only last 128 positions.',
        'array_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
    dump(ROOT/'data/dataset_manifest.json',manifest)
    print('DATA',{'excluded_train_docs':len(excluded),'eligible_fresh_docs':len(eligible),'audit_docs':32,'train_windows':1024},flush=True)


@torch.no_grad()
def cache_split(model,windows,label):
    path=ROOT/'cache'/f'{label}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    hidden=[];loss=[];begin=time.perf_counter()
    for j in range(0,len(windows),2):
        rows=windows[j:j+2];h=model.model(rows[:,:-1],use_cache=False).last_hidden_state[:,-128:]
        hidden.append(h.clone());logits=model.lm_head(h)
        loss.extend(F.cross_entropy(logits.flatten(0,1),rows[:,-128:].flatten(),reduction='none').view(len(rows),-1).mean(1).tolist())
        if (j+2)%256==0:print('CACHE_PROGRESS',label,j+2,flush=True)
    result={'hidden':torch.cat(hidden),'nll':torch.tensor(loss,dtype=torch.float64)}
    torch.save(result,path);print('CACHE',label,len(windows),time.perf_counter()-begin,flush=True);return result


def verify_and_calibrate(model,original,windows,reference):
    before=hashes(model);a,b,u,s,null=basis_analysis()
    identities={'sampling_depths':TIMES,'derivative_sampling_matrix':a.tolist(),'singular_values':s.tolist(),
        'rank_three_step':int(np.linalg.matrix_rank(a[:3])),'rank_two_step':int(np.linalg.matrix_rank(a[3:])),
        'rank_combined':int(np.linalg.matrix_rank(a)),'left_null_vector':null.tolist(),
        'orthogonal_transform':b.tolist(),'orthonormal_sample_basis':u.tolist(),
        'gram_max_abs_error':float(np.max(np.abs((a@b).T@(a@b)-np.eye(4)))),
        'interpretation':'Fixed three-step samples have no expressivity restriction (rank 3). The five combined 3/2-step samples have one affine constraint per tensor coordinate (rank 4).'}
    dump(OUT/'basis_analysis.json',identities)
    targets={};calibration={};initial_equivalence={};native=model(windows['validation'][:1,:-1],use_cache=False).logits.detach()
    for kind in ['dense','raw_curve','orthogonal_curve']:
        banks=make_banks(original,kind)
        with torch.no_grad():
            for n in [4,3,2]:
                set_model(model,original,banks,n);out=model(windows['validation'][:1,:-1],use_cache=False).logits
                if kind=='dense':targets[n]=out.clone()
                difference=float((out-targets[n]).abs().max());assert difference==0,difference
                initial_equivalence[f'{kind}_n{n}']=difference
                if n==4:assert float((out-native).abs().max())==0
        per_n={}
        for n,idx in [(3,torch.tensor([0,1])),(2,torch.tensor([2,3]))]:
            reset_delta(banks);optimizer=torch.optim.AdamW(banks.parameters(),lr=.0001,weight_decay=0)
            optimizer.zero_grad(set_to_none=True)
            set_model(model,original,banks,n);logits=model(windows['train'][idx,:-1],use_cache=False).logits
            with torch.no_grad():lp=F.log_softmax(model.lm_head(reference['hidden'][idx]),-1)
            loss=(lp.exp()*(lp-F.log_softmax(logits,-1))).sum(-1).mean();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(banks.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm)
            optimizer.step();per_n[str(n)]={'loss':float(loss.detach()),'sampled_weight_rms_update':sampled_delta_rms(banks),'gradient_norm':float(norm)}
            del optimizer,logits,loss;reset_delta(banks)
        mean_rms=float(np.mean([r['sampled_weight_rms_update'] for r in per_n.values()]))
        calibration[kind]={'probe_lr':.0001,'per_resolution':per_n,'mean_sampled_weight_rms_update':mean_rms,
            'trainable_values':sum(p.numel() for p in banks.parameters())}
        restore(model,original);del banks;gc.collect()
    target=calibration['dense']['mean_sampled_weight_rms_update']
    for kind,info in calibration.items():
        ratio=target/info['mean_sampled_weight_rms_update'];assert .1<ratio<20
        info['lr_multiplier']=ratio;info['training_peak_lr']=.0001*ratio
    restore(model,original);assert before==hashes(model)
    dump(OUT/'initial_equivalence.json',{'max_logit_differences_to_dense':initial_equivalence,'native_four_step_difference':0.,'original_hashes_unchanged':True})
    dump(OUT/'calibration.json',{'method':'Two one-step training-only probes (N=3 on rows 0/1; N=2 on rows 2/3), reset before each. Match average RMS change across all five sampled weight sets to dense LR 1e-4. Probe updates discarded. Adam is not coordinate invariant; this matches initial update scale, not every future update.', 'arms':calibration})
    print('CALIBRATION',{k:r['training_peak_lr'] for k,r in calibration.items()},flush=True)


if __name__=='__main__':
    setup();select_data();windows=data();model=load_model();original=list(model.model.layers)
    train=cache_split(model,windows['train'],'train');cache_split(model,windows['validation'],'validation')
    verify_and_calibrate(model,original,windows,train)
