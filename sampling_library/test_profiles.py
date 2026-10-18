"""Verify depth profiles reject incompatible metadata without changing saved evidence."""
import copy
import json
import tempfile
from pathlib import Path
from .profiles import INDEX,load_profile
from . import ROOT

def main():
    source=json.loads(INDEX.read_text());entry=source['entries'][0];tests=[]
    with tempfile.TemporaryDirectory(dir=ROOT/'auto-research/results/text-depth-library-20261018-v1/tmp') as tmp:
        index=Path(tmp)/'index.json'
        for field,value,error in [('execution_depth',3,ValueError),('execution_depth',1,ValueError),('base_fingerprint','0'*64,ValueError)]:
            data=copy.deepcopy(source);data['entries'][0][field]=value;index.write_text(json.dumps(data))
            try:load_profile(entry['id'],index)
            except error:tests.append(field+'_'+str(value))
            else:raise AssertionError('incompatible depth profile accepted')
        data=copy.deepcopy(source);data['entries'][0]['model_source']['sha256']='0'*64;index.write_text(json.dumps(data))
        try:load_profile(entry['id'],index)
        except ValueError:tests.append('model_source_hash')
        else:raise AssertionError('wrong source accepted')
    from .runtime import load_router
    model,_=load_profile(entry['id'])
    try:load_router(model,entry['use_case'],entry['tokenizer'],entry['context'])
    except ValueError:tests.append('four_layer_reference_into_two_layer_model')
    else:raise AssertionError('four-layer reference accepted in a two-layer model')
    print('PASS:',len(tests),'depth profile rejection checks')
if __name__=='__main__':main()
