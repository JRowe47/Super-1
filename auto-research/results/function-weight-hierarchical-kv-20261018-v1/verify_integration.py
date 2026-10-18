"""Mechanics evidence only: no language benchmark, selection or deployment."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));H=Path(__file__).resolve().parent
import torch
from sampling_library.profiles import INDEX,load_profile
from sampling_library.inference import CachedSampledCurve,cache_bytes
from sampling_library.hierarchical_kv import HierarchicalKV

def main():
    torch.set_num_threads(1)
    result=subprocess.run([sys.executable,'-B','-m','unittest','sampling_library.test_inference','sampling_library.test_hierarchical_kv'],cwd=ROOT,text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    entries=json.loads(INDEX.read_text())['entries'];records=[]
    for depth in (2,3,4):
        entry=next(e for e in entries if e['execution_depth']==depth and e['seed']==73);model,_=load_profile(entry['id']);original={name:p for name,p in model.named_parameters()};ids=torch.arange(256)[None]
        for recent,levels in [(8,(2,2)),(16,(4,4))]:
            runtime=CachedSampledCurve(model,lambda i,h,d:HierarchicalKV(h,d,recent_capacity=recent,level_capacities=levels))
            with torch.inference_mode():logits,state,t=runtime(ids,positions=True)
            assert torch.isfinite(logits).all();assert all(p is dict(runtime.model.named_parameters())[name] for name,p in original.items())
            bound=depth*2*model.base.tok.embedding_dim*(recent+sum(levels))*4;actual=cache_bytes(state);assert actual<=bound
            records.append({'profile':entry['id'],'depth':depth,'tokens':ids.shape[1],'recent_capacity':recent,'level_capacities':levels,'original_model_parameter_count':sum(p.numel() for p in model.parameters()),'additional_pool_parameters':sum(p.numel() for p in runtime.memory.parameters()),'logical_cache_bytes':actual,'cache_bound_bytes':bound,'exact_KV_bytes':depth*2*model.base.tok.embedding_dim*ids.shape[1]*4,'same_function_weight_and_router_parameter_objects':True,'sampling_coordinate_min':float(t.min()),'sampling_coordinate_max':float(t.max())})
    source_names=['sampling_library/inference.py','sampling_library/hierarchical_kv.py','sampling_library/test_inference.py','sampling_library/test_hierarchical_kv.py','src/hierarchy/experiments/function_weight_wt2_original_transformer_port/model.py']
    receipt={'status':'PASS','kind':'MECHANICS_ONLY','tests':9,'test_output':result.stdout+result.stderr,'trained_profiles_in_tests':9,'cases':records,'source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names},'scope':'Function weights, sampling and layer layout preserved; hierarchy changes attention KV storage/readout only. Pool-learning unit test is synthetic distillation connectivity, not language training or quality promotion. Cache bytes exclude weights/runtime; no total-memory or speed claim.'}
    with (H/'VERIFICATION.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
