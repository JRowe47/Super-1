"""Pinned WikiText-2 raw data; disjoint calibration, selection and test splits."""
from pathlib import Path
import hashlib, json, re, urllib.request
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
REPO='Salesforce/wikitext'
CONFIG='wikitext-2-raw-v1'
COUNTS={'train':128,'validation':32,'test':128}
SEEDS={'train':26317,'validation':26318,'test':26319}
WINDOW=129

def get(url):
    with urllib.request.urlopen(url,timeout=90) as r:return r.read()

def model_dir():
    local=ROOT/'reference_model'
    return local if local.exists() else ROOT.parent/'smollm_first_layer_curve/reference_model'

def main():
    (ROOT/'data').mkdir(exist_ok=True)
    pinned=ROOT/'data/dataset_manifest.json'
    if pinned.exists():
        revision=json.loads(pinned.read_text())['revision']
    else:
        revision=json.loads(get(f'https://huggingface.co/api/datasets/{REPO}/revision/refs%2Fconvert%2Fparquet'))['sha']
    tok=AutoTokenizer.from_pretrained(model_dir(),local_files_only=True)
    # We intentionally tokenize whole articles, then draw independent windows.
    tok.model_max_length=10**9
    manifest={'repository':REPO,'config':CONFIG,'revision':revision,'window_tokens':WINDOW,
              'context_tokens':128,'counts':COUNTS,'seeds':SEEDS,'files':{},'splits':{},
              'sampling':'Uniform without replacement among nonoverlapping 129-token windows within each official split; each window stays inside one article.',
              'held_out_meaning':'Official test split excluded from all calibration and candidate selection; not a claim about original model pretraining contamination.'}
    arrays={};all_titles={}
    for split,count in COUNTS.items():
        path=ROOT/'data'/f'{split}.parquet'
        if not path.exists():
            path.write_bytes(get(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{CONFIG}/{split}/0000.parquet'))
        manifest['files'][path.name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        lines=pq.read_table(path).column('text').to_pylist()
        docs=[];title=None;parts=[]
        for line in lines:
            match=re.match(r'^\s*= ([^=].*?) =\s*$',line)
            if match:
                if parts:docs.append((title or '[untitled]','\n'.join(parts)))
                title=match.group(1).strip();parts=[line]
            else:parts.append(line)
        if parts:docs.append((title or '[untitled]','\n'.join(parts)))
        all_titles[split]={t for t,_ in docs if t!='[untitled]'}
        pool=[];origin=[]
        for docid,(title,text) in enumerate(docs):
            tokens=tok(text,add_special_tokens=False)['input_ids']
            for start in range(0,len(tokens)-WINDOW+1,WINDOW):
                pool.append(tokens[start:start+WINDOW]);origin.append({'document_index':docid,'title':title,'token_offset':start})
        take=np.random.default_rng(SEEDS[split]).choice(len(pool),count,replace=False)
        arrays[split]=np.asarray([pool[i] for i in take],dtype=np.int32)
        manifest['splits'][split]={'document_count':len(docs),'available_windows':len(pool),
                'selected_documents':len({origin[i]['document_index'] for i in take}),
                'selected_windows':[origin[i] for i in take],
                'token_array_sha256':hashlib.sha256(arrays[split].tobytes()).hexdigest()}
        print(split,'documents',len(docs),'selected windows',count,flush=True)
    overlaps={a+'_'+b:sorted(all_titles[a]&all_titles[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    manifest['article_title_overlap']=overlaps
    assert not any(overlaps.values()),overlaps
    np.savez_compressed(ROOT/'data/token_windows.npz',**arrays)
    pinned.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Data preparation complete',revision,flush=True)

if __name__=='__main__':main()
