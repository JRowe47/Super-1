"""Fresh article-held-out phase-3 samples from pinned WikiText-2 raw."""
from pathlib import Path
import hashlib, json, re, urllib.request
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
REVISION='3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e'
COUNTS={'train':1024,'validation':32,'test':128}
SEEDS={'train':30317,'validation':30318,'test':30319}


def model_dir():
    local=ROOT/'reference_model'
    return local if local.exists() else ROOT.parent/'smollm_first_layer_curve/reference_model'


def main():
    old=json.loads((ROOT/'reference/phase2_dataset_manifest.json').read_text())
    tok=AutoTokenizer.from_pretrained(model_dir(),local_files_only=True);tok.model_max_length=10**9
    manifest={'repository':'Salesforce/wikitext','config':'wikitext-2-raw-v1','revision':REVISION,
       'window_tokens':129,'context_tokens':128,'counts':COUNTS,'seeds':SEEDS,'files':{},'splits':{},
       'sampling':'Uniform without replacement among nonoverlapping 129-token windows within articles. Validation/test exclude ALL articles sampled in phase 2.',
       'held_out_meaning':'Unused in phase-2 evaluation and phase-3 training/selection; no claim of absence from original model pretraining.'}
    arrays={};titles={}
    for split,count in COUNTS.items():
        path=ROOT/'data'/f'{split}.parquet'
        prior=ROOT.parent/'smollm_curve_resolution_phase2/data'/f'{split}.parquet'
        if not path.exists() and prior.exists():path=prior
        if not path.exists():
            url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/wikitext-2-raw-v1/{split}/0000.parquet'
            with urllib.request.urlopen(url,timeout=90) as response:path.write_bytes(response.read())
        manifest['files'][path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        docs=[];parts=[];title=None
        for line in pq.read_table(path).column('text').to_pylist():
            match=re.match(r'^\s*= ([^=].*?) =\s*$',line)
            if match:
                if parts:docs.append((title or '[untitled]','\n'.join(parts)))
                title=match.group(1).strip();parts=[line]
            else:parts.append(line)
        if parts:docs.append((title or '[untitled]','\n'.join(parts)))
        titles[split]={t for t,_ in docs if t!='[untitled]'}
        excluded={r['document_index'] for r in old['splits'][split]['selected_windows']} if split!='train' else set()
        pool=[];origins=[]
        for docid,(title,content) in enumerate(docs):
            if docid in excluded:continue
            ids=tok(content,add_special_tokens=False)['input_ids']
            for offset in range(0,len(ids)-128,129):
                pool.append(ids[offset:offset+129]);origins.append({'document_index':docid,'title':title,'token_offset':offset})
        take=np.random.default_rng(SEEDS[split]).choice(len(pool),count,replace=False)
        arrays[split]=np.asarray([pool[i] for i in take],dtype=np.int32)
        selected=[origins[i] for i in take]
        assert not (set(r['document_index'] for r in selected)&excluded)
        manifest['splits'][split]={'document_count':len(docs),'available_windows':len(pool),
           'excluded_phase2_document_indices':sorted(excluded),'selected_documents':len({r['document_index'] for r in selected}),
           'selected_windows':selected,'token_array_sha256':hashlib.sha256(arrays[split].tobytes()).hexdigest()}
        print(split,count,'windows;',manifest['splits'][split]['selected_documents'],'articles;',len(excluded),'previous articles excluded',flush=True)
    overlaps={a+'_'+b:sorted(titles[a]&titles[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    assert not any(overlaps.values());manifest['article_title_overlap']=overlaps
    np.savez_compressed(ROOT/'data/token_windows.npz',**arrays)
    (ROOT/'data/dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
