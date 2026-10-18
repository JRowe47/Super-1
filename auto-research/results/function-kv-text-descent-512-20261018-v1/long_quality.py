"""Fresh continuous8K byte-text diagnostic with the selected512-context model."""
import hashlib,json,math,statistics,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
import run as pilot
H=Path(__file__).resolve().parent;O=H/'execution'
def main():
    pilot.setup();p=json.loads((H/'LONG_QUALITY_PROTOCOL.json').read_text());freeze=json.loads((O/'FREEZE.json').read_text());out=O/'long-quality';out.mkdir(exist_ok=False)
    pilot.save(out/'START.json',{'freeze_sha256':pilot.sha(O/'FREEZE.json'),'protocol_sha256':pilot.sha(H/'LONG_QUALITY_PROTOCOL.json'),'selected_capacity':freeze['selected_capacity'],'corpus_sha256':pilot.sha(H/'corpus.txt')})
    a,b=p['byte_region'];tokens=torch.tensor(list((H/'corpus.txt').read_bytes()[a:b]),dtype=torch.long).unsqueeze(0);assert tokens.shape[1]==p['context']+1
    rows=[]
    for seed in pilot.C['seeds']:
        base=next(r for r in freeze['bases'] if r['arm']=='curve' and r['seed']==seed)
        for arm in ['exact',*pilot.C['candidates']]:
            model=pilot.build('curve',seed);model.load_state_dict(torch.load(O/base['checkpoint'],weights_only=True)['model']);m=pilot.runtime(model,arm).eval()
            if arm!='exact':
                choice=next(r for r in freeze['hierarchy'] if r['arm']==arm and r['seed']==seed);m.memory.load_state_dict(torch.load(O/choice['checkpoint'],weights_only=True)['model'])
            metrics=[];start=time.process_time()
            for repeat in range(2):
                cache=None;loss=tail=0.;digest=hashlib.sha256()
                with torch.inference_mode():
                    for i in range(0,p['context'],p['chunk']):
                        end=min(i+p['chunk'],p['context']);z,cache=m(tokens[:,i:end],cache);target=tokens[:,i+1:end+1]
                        ce=F.cross_entropy(z.flatten(0,1),target.flatten(),reduction='none');assert torch.isfinite(ce).all();loss+=ce.sum().item()
                        if end>p['context']//2:tail+=ce[max(0,p['context']//2-i):].sum().item()
                        digest.update(z.numpy().tobytes())
                metrics.append({'ce':loss/p['context'],'tail_ce':tail/(p['context']//2),'logits_sha256':digest.hexdigest(),'cache_bytes':pilot.cache_bytes(cache)})
            assert metrics[0]==metrics[1]
            row={'seed':seed,'arm':arm,'base_checkpoint':base['checkpoint'],'base_sha256':base['sha256'],'metrics':metrics[0],'duplicate_exact':True,'cpu_seconds':time.process_time()-start};rows.append(row);pilot.save(out/f'{arm}-{seed}.json',row);print(json.dumps(row),flush=True)
    contrasts={}
    for arm in pilot.C['candidates']:
        for key in ('ce','tail_ce'):
            gaps=[next(r['metrics'][key] for r in rows if r['arm']==arm and r['seed']==s)-next(r['metrics'][key] for r in rows if r['arm']=='exact' and r['seed']==s) for s in pilot.C['seeds']];mean=statistics.mean(gaps);radius=pilot.C['t95_df2']*statistics.stdev(gaps)/math.sqrt(len(gaps))
            contrasts[arm+'_'+key]={'gaps':gaps,'mean':mean,'paired_t95':[mean-radius,mean+radius],'geometric_ppl_ratio':math.exp(mean),'pass':mean+radius<=math.log1p(p['quality_ppl_tolerance']) and max(gaps)<=math.log1p(p['quality_ppl_tolerance'])}
    selected=freeze['selected_capacity'];pilot.save(out/'RESULT.json',{'rows':rows,'contrasts':contrasts,'selected_capacity':selected,'selected_quality_gate_pass':all(contrasts[selected+'_'+key]['pass'] for key in ('ce','tail_ce')),'scope':p['scope']})
if __name__=='__main__':main()
