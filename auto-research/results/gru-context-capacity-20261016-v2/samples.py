"""Unfiltered illustrations from three 2K endpoints; not a quality metric."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import experiment as ex
import torch
from tokenizers import Tokenizer
ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);args=ap.parse_args()
p,_,_,_=ex.inputs(args);summary=json.loads((ROOT/'SUMMARY.json').read_text())
rows=summary['rows'];chosen=[]
for arm in ('mingru_linear','native','nanogpt'):
    available=[r for r in rows if r['arm']==arm and r['context']==2048]
    if available:chosen.append(min(available,key=lambda r:r['ce']))
tok=Tokenizer.from_file(str(args.data_dir/'tokenizer.json'))
prompts=['The purpose of a scientific experiment is','A computer network']
records=[]
with torch.no_grad():
    for r in chosen:
        q=next(q for q in p['runs'] if q['run_id']==r['run_id']);m=ex.build(q,p)
        cp=torch.load(ROOT/'execution'/q['run_id']/'checkpoint.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['model']);m.eval()
        for i,prompt in enumerate(prompts):
            torch.manual_seed(101616+i);ids=torch.tensor([tok.encode(prompt).ids],dtype=torch.long);generated=[]
            if q['arm']!='nanogpt':
                out=m(ids);state=out.final_state;logits=out.logits[:,-1]
            for step in range(64):
                if q['arm']=='nanogpt':logits=m(ids)[0][:,-1]
                values,indices=torch.topk(logits/.8,50,dim=-1)
                choice=torch.multinomial(torch.softmax(values,-1),1);token=indices.gather(1,choice)
                generated.append(int(token));ids=torch.cat((ids,token),1)
                if q['arm']!='nanogpt':logits,state,_=m.step(token[:,0],state)
            records.append(dict(run_id=q['run_id'],prompt=prompt,seed=101616+i,temperature=.8,top_k=50,
                generated_tokens=generated,continuation=tok.decode(generated)))
ex.save(ROOT/'SAMPLES.json',dict(selection='Best final-CE 2K minGRU plus ordinary GRU and nanoGPT 2K endpoints; chosen after results only for illustrations.',
    all_samples_retained=True,records=records))
