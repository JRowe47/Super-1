"""Independently reload every library profile and reproduce confirmation logits."""
import hashlib,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from sampling_library.hierarchy_profiles import load_hierarchy_profile
from sampling_library import artifact

def verify(directory):
    library=directory/'LOADABLE_LIBRARY.json';data=json.loads(library.read_text());records=[]
    result=json.loads((directory/'execution/RESULT.json').read_text())
    for entry in data['entries']:
        runtime,evidence=load_hierarchy_profile(entry['id'],library);config=json.loads(artifact(entry['config']).read_text());a,b=config['text_regions']['confirmation'];d=torch.tensor(list((directory/'corpus.txt').read_bytes()[a:b]),dtype=torch.long)
        ix=torch.arange(config['confirmation_examples'])[:,None]*(config['context']+1)+torch.arange(config['context'])[None,:];x=d[ix].contiguous();y=d[ix+1].contiguous();digest=hashlib.sha256();loss=0.
        with torch.no_grad():
            for i in range(0,len(x),config['batch']):
                z,_=runtime(x[i:i+config['batch']]);target=y[i:i+config['batch']];loss+=F.cross_entropy(z.flatten(0,1),target.flatten(),reduction='sum').item();digest.update(z.numpy().tobytes())
        row=next(r for r in result['rows'] if r.get('variant')=='selected' and r['checkpoint']==Path(entry['pool_checkpoint']['path']).name)
        assert digest.hexdigest()==row['confirmation']['logits_sha256'] and loss/y.numel()==row['confirmation']['ce']
        records.append({'id':entry['id'],'confirmation_logits_sha256':digest.hexdigest(),'ce':loss/y.numel(),'exact_replay':True})
    receipt={'status':'PASS','profiles':records,'loader_sha256':hashlib.sha256((ROOT/'sampling_library/hierarchy_profiles.py').read_bytes()).hexdigest()}
    with (directory/'execution/LIBRARY_REPLAY.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'directory':str(directory),'profiles':len(records),'status':'PASS'}))
if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for argument in sys.argv[1:]:verify(Path(argument))
