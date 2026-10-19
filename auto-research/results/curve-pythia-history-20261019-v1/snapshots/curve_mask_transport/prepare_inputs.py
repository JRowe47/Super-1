"""Freeze new evaluation and mask-selection inputs; preserve paired replay stream."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
def main(source,checkpoints):
    rng=np.random.default_rng(2026101720)
    tok=AutoTokenizer.from_pretrained(checkpoints/'tokenizer',local_files_only=True)
    p=ROOT/'provenance'
    old=json.loads((p/'retention_data_previous_selection.json').read_text())
    previous=json.loads((p/'retention_data_phase_selection.json').read_text())
    z=np.load(p/'retention_data_phase_blocks.npz')
    data={'train':z['train'].copy(),'replay':z['replay'].copy()}
    excluded={r['row'] for r in old['validation']+previous['validation']}
    texts=pq.read_table(source/'validation.parquet',columns=['text']).column(0).to_pylist()
    candidates=rng.permutation([i for i,t in enumerate(texts) if len(t)>600 and i not in excluded])
    chosen=[];meta=[]
    for i in candidates:
        ids=tok(texts[i],add_special_tokens=False).input_ids
        if len(ids)<96:continue
        offset=int(rng.integers(0,len(ids)-95));chosen.append(ids[offset:offset+96])
        meta.append(dict(row=int(i),offset=offset,row_text_sha256=hashlib.sha256(texts[i].encode()).hexdigest()))
        if len(chosen)==48:break
    assert len(chosen)==48
    data['validation']=np.array(chosen,np.int64)
    prior=json.loads((p/'retention_results_probe_inputs.json').read_text())
    earlier=json.loads((p/'retention_provenance_probe_inputs.json').read_text())
    oldtokens=set(json.loads((p/'retention_data_prior_probe_token_ids.json').read_text()))
    for d in [prior,earlier]:
        for rows in d['sets'].values():
            for row in rows:oldtokens.update(row['prefix'])
    pool=rng.permutation([i for i in range(100,50000) if i not in oldtokens]);offset=0
    sets={'calibration':prior['sets']['calibration']}
    for split,counts in [('mask_selection',[(16,8),(32,8),(48,8)]),('retention',[(16,16),(32,16),(48,16),(64,3),(96,3)])]:
        rows=[]
        for n,count in counts:
            for j in range(count):
                rows.append(dict(id=len(rows),n=n,prefix=pool[offset:offset+n].tolist(),order=rng.permutation(n-1).tolist()));offset+=n
        sets[split]=rows
    a={i for row in sets['mask_selection'] for i in row['prefix']};b={i for row in sets['retention'] for i in row['prefix']}
    assert not(a&b) and not((a|b)&oldtokens)
    assert not {r['row'] for r in meta}&excluded
    (ROOT/'results/probe_inputs.json').write_text(json.dumps(dict(sets=sets,prose=prior['prose'],prose_ids=prior['prose_ids']),indent=2)+'\n')
    np.savez_compressed(ROOT/'data/phase_blocks.npz',**data)
    (ROOT/'data/evaluation_selection.json').write_text(json.dumps(meta,indent=2)+'\n')
    checks=dict(new_validation_rows=48,excluded_previous_validation_rows=len(excluded),fresh_copy_selection_and_test_disjoint=True,prior_copy_ids_excluded=True,adaptation_replay_equal_previous=all(np.array_equal(data[k],z[k]) for k in ['train','replay']),copy_counts={k:len(v) for k,v in sets.items()})
    (ROOT/'results/input_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-data',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);a=p.parse_args();main(a.source_data,a.checkpoints)
