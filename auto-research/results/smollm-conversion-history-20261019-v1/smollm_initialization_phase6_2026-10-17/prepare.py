"""Reserve unseen audit articles, cache the teacher, and lock the protocol."""
import gc,hashlib,json,re,time,urllib.request
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from common import ROOT,OUT,SEED,UPDATES,PEAK_LR,KEEP,dump,setup,model_dir,load_model,data,hashes
from models import candidate,install,restore,lift,install_curve,SAMPLE_MATRIX,RIGHT_INVERSE,TIMES
from evaluation import evaluate

def select_data():
    if (ROOT/'data/token_windows.npz').exists():return
    previous=json.loads((ROOT/'reference/phase5_dataset_manifest.json').read_text())
    with np.load(ROOT/'reference/phase5_token_windows.npz') as a:
        arrays={'train':a['train'].copy(),'validation':a['validation'].copy()}
    previous_audit={r['document_index'] for r in previous['audit_windows_manifest']}
    excluded=set(previous['excluded_train_document_indices'])|previous_audit
    path=ROOT/'cache/train.parquet'
    if not path.exists():
        url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{previous["revision"]}/wikitext-2-raw-v1/train/0000.parquet'
        with urllib.request.urlopen(url,timeout=90) as r:path.write_bytes(r.read())
    sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==previous['train_parquet_sha256']
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
        ids=tok(text,add_special_tokens=False)['input_ids'];starts=list(range(0,len(ids)-256,257))
        if len(starts)>=2:eligible[i]=(title,ids,starts)
    rng=np.random.default_rng(SEED+1);chosen=rng.choice(sorted(eligible),32,replace=False)
    audit=[];origins=[]
    for doc in chosen:
        title,ids,starts=eligible[int(doc)]
        for offset in rng.choice(starts,2,replace=False):
            audit.append(ids[int(offset):int(offset)+257]);origins.append({'document_index':int(doc),'title':title,'token_offset':int(offset),'official_split':'train'})
    arrays['audit_long']=np.asarray(audit,dtype=np.int32);arrays['audit']=arrays['audit_long'][:,128:].copy()
    assert not {r['document_index'] for r in origins}&excluded
    np.savez_compressed(ROOT/'data/token_windows.npz',**arrays)
    manifest={'repository':previous['repository'],'config':previous['config'],'revision':previous['revision'],
        'train_parquet_sha256':sha,'train_windows':len(arrays['train']),'validation_windows':len(arrays['validation']),
        'audit_windows':64,'audit_documents':32,'audit_sampling_seed':SEED+1,
        'audit_source':'Another 32 previously unused official-train WikiText-2 articles, reserved before optimization and evaluation. Not the official test partition.',
        'audit_exclusions':'Every official-train article used for phase-2/3 training (phase-4/5 training is a subset), plus all phase-5 audit articles.',
        'excluded_document_indices':sorted(excluded),'previous_audit_document_indices':sorted(previous_audit),
        'eligible_audit_documents':len(eligible),'audit_windows_manifest':origins,
        'train_windows_manifest':previous['train_windows_manifest'],'validation_windows_manifest':previous['validation_windows_manifest'],
        'long_context':'Expose 128 extra preceding tokens but score the same final 128 targets as the short audit.',
        'pretraining_limit':'No claim that these articles were absent from the reference model\'s original pretraining.',
        'array_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
    dump(ROOT/'data/dataset_manifest.json',manifest)
    print('DATA',{'excluded_docs':len(excluded),'eligible_docs':len(eligible),'audit_docs':32},flush=True)

@torch.no_grad()
def cache_split(model,windows,label):
    path=ROOT/'cache'/f'{label}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    hidden=[];loss=[];begin=time.perf_counter()
    for start in range(0,len(windows),2):
        rows=windows[start:start+2]
        h=model.model(rows[:,:-1],use_cache=False).last_hidden_state[:,-128:]
        hidden.append(h.clone());logits=model.lm_head(h)
        loss.extend(F.cross_entropy(logits.flatten(0,1),rows[:,-128:].flatten(),reduction='none').view(len(rows),-1).mean(1).tolist())
        if (start+2)%256==0:print('CACHE_PROGRESS',label,start+2,flush=True)
    result={'hidden':torch.cat(hidden),'nll':torch.tensor(loss,dtype=torch.float64)}
    torch.save(result,path);print('CACHE',label,len(windows),time.perf_counter()-begin,flush=True);return result

def protocol():
    indices=torch.randint(0,1024,(UPDATES,2),generator=torch.Generator().manual_seed(SEED)).numpy()
    np.save(ROOT/'data/training_indices.npy',indices)
    spec={'phase':6,'model':'HuggingFaceTB/SmolLM2-135M','model_revision':'93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
        'question':'Does retained-native initialization recover reduced-depth quality better than interpolated initialization under matched longer distillation?',
        'windows_zero_indexed':[[4,5,6,7],[12,13,14,15]],'steps_per_window':3,'decoder_blocks':28,
        'arms':['interpolated','retained'],'retained_offsets':KEEP,'removed_original_blocks':[5,13],
        'retention_policy':'Predetermined endpoints plus rounded-up middle of linspace(0,3,3), giving offsets [0,2,3]. No pruning search.',
        'interpolated_initialization':'Phase-4 selected Hermite curve samples at [1/6, 1.5, 17/6]. These slopes have earlier training history.',
        'primary_residual_scale':1.,'scaling_rationale':'Both primary arms use ordinary native transformer residual strength. Separately score the old 4/3 multiplier at initialization on validation; no scale selection using the audit.',
        'trainable_parameters':21240576,'frozen':'All original knots are immutable. Only six cloned/reconstructed blocks train; other 22 blocks, embeddings, output head and final norm remain frozen.',
        'updates_per_arm':UPDATES,'batch_sequences':2,'context':128,'seed':SEED,
        'input_tokens_per_arm':UPDATES*256,'data_pool_input_tokens':1024*128,
        'phase5_comparison':'Each phase-5 dense resolution had 300 updates. This phase trains N=3 only for 1800 updates (6x per-resolution exposure). It also changes residual scale and schedule, so phase-5 to phase-6 improvement is not isolated to duration alone.',
        'objective':'Full-model teacher-to-student next-token KL at temperature 1; no intermediate-state loss.',
        'optimizer':{'name':'AdamW','lr':PEAK_LR,'betas':[.9,.999],'weight_decay':0.,'gradient_clip_norm':1.,'warmup_updates':60,'schedule':'cosine to 20% of peak'},
        'matched':'Both arms use identical ordered batches, exposure, trainable count, objective, optimizer and residual convention. No extra LR sweep.',
        'selection':'Check every 300 updates including 0; minimum mean KL on first 8 validation windows. Gate on all 32 validation windows: delta NLL <= .05 and KL <= .05.',
        'audit':'64 windows / 8192 targets / 32 new document-held-out official-train articles, excluding all phase-5 audit articles. No audit-driven training or selection.',
        'audit_candidates':['initial_interpolated','initial_retained','phase5_dense','trained_interpolated','trained_retained'],
        'interaction_ablations':'Each trained candidate at each window alone, using original weights in the other window; score the same fresh audit.',
        'long_context':'Phase-5 dense and both trained candidates at context 256, score the same final 128 targets.',
        'curve_lift':'Three cardinal functions l(t)=a(t) A3+ vanish at every original knot and form I3 at the three fitted samples. Store target-minus-base tensors. Unit residual strength; original four-step and fitted three-step schedules only.',
        'conversion_checks':'All sampled tensors; end-to-end logits on 8 audit windows; exact original four-step logits; DynamicCache; selected-block reload; ordinary safetensors model reload.',
        'limits':'One seed, two four-block regions, fixed depth and short text windows. No claim about N=2, arbitrary unseen depths, input-conditioned GSA, original-pretraining leakage or state/cache asymptotics.'}
    dump(OUT/'protocol.json',spec);return spec

@torch.no_grad()
def initial_checks(model,original,windows):
    before=hashes(model);ids=windows['validation'][:1,:-1]
    baseline=model(ids,use_cache=False).logits
    b=candidate(original,'retained');install(model,original,b);copied=model(ids,use_cache=False).logits
    kept=[block for i,block in enumerate(original) if i not in [5,13]]
    model.model.layers=torch.nn.ModuleList(kept)
    for i,block in enumerate(kept):block.self_attn.layer_idx=i
    direct=model(ids,use_cache=False).logits
    retained_difference=float((copied-direct).abs().max());assert retained_difference==0
    banks=lift(original,b);install_curve(model,original,banks,4)
    original_difference=float((model(ids,use_cache=False).logits-baseline).abs().max());assert original_difference==0
    install_curve(model,original,banks,3)
    curve_difference=float((model(ids,use_cache=False).logits-copied).abs().max());assert curve_difference<.001
    restore(model,original);assert before==hashes(model)
    dump(OUT/'initial_checks.json',{'retained_copy_vs_direct_pruning_max_logit_difference':retained_difference,
        'four_step_vs_original_max_logit_difference':original_difference,'three_step_curve_vs_native_max_logit_difference':curve_difference,
        'three_sample_basis_rank':int(np.linalg.matrix_rank(SAMPLE_MATRIX)),
        'sample_cardinal_identity_error':float(np.abs(SAMPLE_MATRIX@RIGHT_INVERSE-np.eye(3)).max()),
        'original_parameter_hashes_unchanged':True,'trainable_values':sum(p.numel() for p in b.parameters())})

def main():
    setup();select_data();protocol();windows=data();model=load_model();original=list(model.model.layers)
    initial_checks(model,original,windows)
    cache_split(model,windows['train'],'train');reference=cache_split(model,windows['validation'],'validation')
    cache_split(model,windows['audit'],'audit');cache_split(model,windows['audit_long'],'audit_long')
    path=OUT/'initial_validation.json';rows=json.loads(path.read_text()) if path.exists() else [];done={r['id'] for r in rows}
    for kind in ['interpolated','retained']:
        for scale in [1.,4/3]:
            label=f'{kind}_scale_{scale:.6f}'
            if label in done:continue
            blocks=candidate(original,kind,scale=scale);install(model,original,blocks)
            m=evaluate(model,windows['validation'],reference)
            rows.append({'id':label,'initialization':kind,'residual_scale':scale,'metrics':m});dump(path,rows)
            print('INITIAL',label,m['perplexity_ratio_to_original'],m['mean_kl'],flush=True)
            restore(model,original);del blocks;gc.collect()
    print('PREPARATION_COMPLETE',flush=True)

if __name__=='__main__':main()
