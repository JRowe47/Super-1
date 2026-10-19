"""Freeze disjoint replay and held-out inputs before training."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def main(source,checkpoints):
    rng=np.random.default_rng(2026101718);tok=AutoTokenizer.from_pretrained(checkpoints/'tokenizer',local_files_only=True)
    previous=json.loads((ROOT/'data/previous_selection.json').read_text());old=np.load(ROOT/'data/lm_token_blocks.npz');blocks={'train':old['train'].copy()};meta={}
    for split,key in [('train','replay'),('validation','validation')]:
        rows=pq.read_table(source/f'{split}.parquet',columns=['text']).column(0).to_pylist();excluded={r['row'] for r in previous[split]}
        picks=rng.permutation([i for i,t in enumerate(rows) if len(t)>600 and i not in excluded]);selected=[];m=[]
        for i in picks:
            ids=tok(rows[i],add_special_tokens=False).input_ids
            if len(ids)<96:continue
            start=int(rng.integers(0,len(ids)-95));selected.append(ids[start:start+96]);m.append(dict(row=int(i),offset=start,row_text_sha256=hashlib.sha256(rows[i].encode()).hexdigest()))
            if len(selected)==24:break
        assert len(selected)==24;blocks[key]=np.array(selected,np.int64);meta[key]=m
    prior=json.loads((ROOT/'provenance/probe_inputs.json').read_text())
    excluded=set(json.loads((ROOT/'data/prior_probe_token_ids.json').read_text()))
    for rows in prior['sets'].values():
        for r in rows:excluded.update(r['prefix'])
    pool=rng.permutation([i for i in range(100,50000) if i not in excluded]);off=0;sets={}
    for split in ['calibration','retention']:
        seqs=[]
        for n,count in ([(16,8),(32,8),(48,8)] if split=='calibration' else [(16,8),(32,8),(48,8),(64,3),(96,3)]):
            for j in range(count):
                seqs.append(dict(id=len(seqs),n=n,prefix=pool[off:off+n].tolist(),order=rng.permutation(n-1).tolist()));off+=n
        sets[split]=seqs
    ca={i for r in sets['calibration'] for i in r['prefix']};te={i for r in sets['retention'] for i in r['prefix']};assert not ca&te and not(ca|te)&excluded
    prose=json.loads((ROOT/'data/retention_prose.json').read_text());dump(ROOT/'results/probe_inputs.json',dict(sets=sets,prose={'retention':prose['texts']},prose_ids={'retention':prose['ids']}))
    np.savez_compressed(ROOT/'data/phase_blocks.npz',**blocks);dump(ROOT/'data/phase_selection.json',meta)
    dump(ROOT/'results/input_checks.json',dict(replay_test_copy_tokens_disjoint=True,previous_probe_tokens_excluded=True,replay_adaptation_rows_disjoint=True,new_validation_rows_disjoint=True,adaptation_equal_previous=bool(np.array_equal(blocks['train'],old['train'])),counts={k:list(v.shape) for k,v in blocks.items()},copy_sequences={k:len(v) for k,v in sets.items()}))
    print('Frozen replay/evaluation inputs; all separation checks passed.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-data',type=Path,default=ROOT/'data');p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');a=p.parse_args();main(a.source_data,a.checkpoints)
