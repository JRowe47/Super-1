"""Exploratory longer-delay evaluation, registered after main ceiling observed.

No retraining, selection, or change to the main decision rule. Chunked recurrent
execution bounds activation memory; state is never reset between chunks.
"""
import hashlib,json,time
import numpy as np
import torch
from torch.nn import functional as F
from experiment import ROOT,save,sha
from model import Model,ARMS
from tasks import episode

@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=json.loads((ROOT/'EXPLORATORY_PROTOCOL.json').read_text())
    assert p['source_sha256']==sha(ROOT/'stress.py')
    rows=[];start=time.perf_counter();cpu=time.process_time()
    for seed in p['model_seeds']:
        models={}
        for arm in ARMS:
            checkpoint=ROOT/'main'/f'{arm}_s{seed}'/'final.pt'
            assert sha(checkpoint)==p['checkpoints'][f'{arm}_s{seed}']
            saved=torch.load(checkpoint,weights_only=True);model=Model(arm);model.load_state_dict(saved['model']);model.eval();models[arm]=model
        for delay in p['delays']:
            acc={arm:[] for arm in ARMS};loss={arm:[] for arm in ARMS};digest=hashlib.sha256()
            for base in p['evaluation_seeds']:
                for k in range(4):
                    x,y,_=episode(base+100*k,batch=32,delay=delay,mode='retention')
                    digest.update(x.numpy().tobytes());digest.update(y.numpy().tobytes())
                    for arm,m in models.items():
                        state=None;tail=[]
                        for left in range(0,x.shape[1],p['chunk_size']):
                            logits,state=m(x[:,left:left+p['chunk_size']],state)
                            # Preserve four final outputs even for a short last chunk.
                            tail.append(logits[:,-4:]);tail=[torch.cat(tail,1)[:,-4:]]
                        logits=tail[0];labels=y[:,-4:]
                        acc[arm].extend((logits.argmax(-1)==labels).float().mean(1).tolist())
                        loss[arm].extend(F.cross_entropy(logits.reshape(-1,8),labels.reshape(-1),reduction='none').reshape(-1,4).mean(1).tolist())
            for arm in ARMS:
                row=dict(arm=arm,seed=seed,mode='retention',delay=delay,episodes=len(acc[arm]),accuracy=float(np.mean(acc[arm])),
                         nats_per_query=float(np.mean(loss[arm])),episode_accuracy=acc[arm],data_sha256=digest.hexdigest())
                rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='episode_accuracy'}),flush=True)
            save(ROOT/'EXPLORATORY_PROGRESS.json',dict(rows=rows,wall_seconds=time.perf_counter()-start))
    summaries={arm:[dict(delay=delay,accuracy_mean=float(np.mean([r['accuracy'] for r in rows if r['arm']==arm and r['delay']==delay])),
                   accuracy_per_seed=[r['accuracy'] for r in rows if r['arm']==arm and r['delay']==delay]) for delay in p['delays']] for arm in ARMS}
    save(ROOT/'EXPLORATORY.json',dict(status='COMPLETE',registered_after_main_results=True,no_retraining=True,
         rows=rows,summary=summaries,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
if __name__=='__main__':main()
