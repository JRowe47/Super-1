"""Exercise real checkpoint reload, incompatibility rejection and index updates."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
import torch
from . import DEFAULT_INDEX, ROOT, artifact, fingerprint, load_router, read, register, resolve

def main():
    h=ROOT/'auto-research/results/text-sampling-library-20261018-v1'
    spec=importlib.util.spec_from_file_location('library_study',h/'run.py');study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study);study.setup()
    tests=[]
    def rejected(name,fn,error):
        try:fn()
        except error:tests.append(name)
        else:raise AssertionError(name+' unexpectedly succeeded')
    for entry in read()['entries']:
        if not entry['active']:continue
        ref=next(r for r in study.references() if r['seed']==entry['seed'])
        expected=study.model(ref,artifact(entry['checkpoint']));m=study.model(ref)
        with torch.no_grad():
            for p in m.router.parameters():p.zero_()
        loaded=load_router(m,entry['use_case'],entry['tokenizer'],entry['context'])
        assert loaded['id']==entry['id']
        x=study.evaluation('selection')[0][:study.C['batch']]
        with torch.no_grad():assert torch.equal(m(x),expected(x))
        tests.append('exact_reload_seed_'+str(entry['seed']))
        with torch.no_grad():next(m.base.parameters()).add_(1)
        rejected('different_backbone_seed_'+str(entry['seed']),lambda:load_router(m,entry['use_case'],entry['tokenizer'],entry['context']),LookupError)
    entry=next(e for e in read()['entries'] if e['active'])
    for use_case,tokenizer,context in [('unseen/domain',entry['tokenizer'],entry['context']),(entry['use_case'],'other-tokenizer',entry['context']),(entry['use_case'],entry['tokenizer'],entry['context']*2)]:
        rejected('incompatible_'+use_case+'_'+tokenizer+'_'+str(context),lambda:resolve(use_case,entry['base_fingerprint'],tokenizer,context),LookupError)
    with tempfile.TemporaryDirectory(dir=h/'tmp') as tmp:
        index=Path(tmp)/'index.json';register(entry,index)
        rejected('duplicate_id',lambda:register(entry,index),ValueError)
        bad=copy.deepcopy(entry);bad['id']='broken';bad['checkpoint']['sha256']='0'*64
        rejected('tampered_artifact',lambda:register(bad,index),ValueError)
        bad=copy.deepcopy(entry);bad['id']='unconfirmed';bad['status']='candidate'
        rejected('unconfirmed_promotion',lambda:register(bad,index),ValueError)
        replacement=copy.deepcopy(entry);replacement['id']='replacement';register(replacement,index)
        assert resolve(entry['use_case'],entry['base_fingerprint'],entry['tokenizer'],entry['context'],index)['id']=='replacement'
        assert len(read(index)['entries'])==2 and sum(e['active'] for e in read(index)['entries'])==1
        tests.append('preserve_old_reference_on_replacement')
    study.save(h/'execution/LIBRARY_VERIFICATION.json',{'status':'PASS','checks':tests})
    print('PASS:',len(tests),'library checks')
if __name__=='__main__':main()
