"""Pinned source selection; saved token IDs are the experiment's exact inputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
def main(checkpoints):
    rng=np.random.default_rng(2026101709);tok=AutoTokenizer.from_pretrained(checkpoints/'tokenizer',local_files_only=True)
    selected={};meta={}
    for split,count in [('train',48),('validation',24)]:
        rows=pq.read_table(ROOT/'data'/f'{split}.parquet',columns=['text']).column(0).to_pylist()
        candidates=rng.permutation([i for i,t in enumerate(rows) if len(t)>600]);blocks=[];source=[]
        for i in candidates:
            ids=tok(rows[i],add_special_tokens=False).input_ids
            if len(ids)<96:continue
            start=int(rng.integers(0,len(ids)-96+1));blocks.append(ids[start:start+96]);source.append(dict(row=int(i),offset=start,row_text_sha256=hashlib.sha256(rows[i].encode()).hexdigest()))
            if len(blocks)==count:break
        assert len(blocks)==count;selected[split]=np.array(blocks,dtype=np.int64);meta[split]=source
    np.savez_compressed(ROOT/'data/lm_token_blocks.npz',**selected)
    (ROOT/'data/lm_selection.json').write_text(json.dumps(meta,indent=2)+'\n')
    excluded=set(json.loads((ROOT/'data/prior_probe_token_ids.json').read_text()));pool=rng.permutation([i for i in range(100,50000) if i not in excluded]);off=0;seqs=[]
    for n in [16,32,48]:
        for j in range(8):
            seqs.append(dict(id=len(seqs),n=n,prefix=pool[off:off+n].tolist(),order=rng.permutation(n-1).tolist()));off+=n
    prose=json.loads((ROOT/'data/retention_prose.json').read_text())
    (ROOT/'results/probe_inputs.json').write_text(json.dumps(dict(sets={'retention':seqs},prose={'retention':prose['texts']},prose_ids={'retention':prose['ids']}),indent=2)+'\n')
    print('Prepared 48 training blocks, 24 validation blocks, 24 new copying sequences, 8 prose passages.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');main(p.parse_args().checkpoints)
