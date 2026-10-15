
"""Acceptance check of recovered historical hierarchy snapshot and checkpoint layout."""
import argparse,hashlib,importlib.util,json,pathlib,sys,time,types
def require(ok,message):
    if not ok: raise RuntimeError(message)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--package-root",type=pathlib.Path,required=True)
    p.add_argument("--hierarchy-checkpoint",type=pathlib.Path,required=True)
    p.add_argument("--confirm-synthetic-only",action="store_true",required=True)
    args=p.parse_args();root=args.package_root;identity=json.loads((root/"provenance/source_identity.json").read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=[]
    for rel,expected in identity["historical_hierarchy"]["included_snapshot"].items():
        file=root/"source/historical_hierarchy"/rel
        require(sha(file)==expected,"Historical snapshot source mismatch "+rel);sources.append({"path":rel,"sha256":expected})
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    package=types.ModuleType("representation_experiments")
    package.__path__=[str(root/"source/historical_hierarchy/representation_experiments")]
    sys.modules[package.__name__]=package
    load("representation_experiments.common",root/"source/historical_hierarchy/representation_experiments/common.py")
    models=load("representation_experiments.predictive_models",root/"source/historical_hierarchy/representation_experiments/predictive_models.py")
    rows=[]
    for seed in (73,191,307):
        torch.manual_seed(seed)
        # Small synthetic fixture exercises fast, segment and document updates.
        model=models.HierarchicalDistributionStudent(64,16,16,4,2).double().cpu()
        x=torch.randint(64,(2,16));target=torch.randint(64,(2,16))
        with torch.no_grad():
            full=model(x)
            changed=x.clone();changed[:,8:]=(changed[:,8:]+1)%64;altered=model(changed)
            state=None;parts=[]
            for j in range(16):
                logits,state,_=model.step(x[:,j],state);parts.append(logits)
            step_logits=torch.stack(parts,1)
        require(torch.allclose(full.logits,step_logits,atol=1e-9,rtol=1e-9),"Historical forward vs stateful step mismatch")
        require(torch.equal(full.logits[:,:8],altered.logits[:,:8]),"Historical causal prefix violation")
        for key in state:
            require(torch.equal(state[key],full.final_state[key]) if isinstance(state[key],torch.Tensor) else state[key]==full.final_state[key],"Historical state mismatch "+key)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        loss=F.cross_entropy(model(x).logits.reshape(-1,64),target.reshape(-1));loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"Historical missing/nonfinite gradient")
        for name in ("fast_cell","segment_cell","document_cell"):
            require(any(p.grad.abs().sum()>0 for p in getattr(model,name).parameters()),"Unexercised hierarchy scale "+name)
        before=model.embedding.weight.detach().clone();optimizer.step()
        require(not torch.equal(before,model.embedding.weight),"Historical optimizer update absent")
        rows.append({"seed":seed,"forward_vs_step_max_abs":float((full.logits-step_logits).abs().max()),"all_three_gru_scales_gradient_tested":True,"state_and_counters_match":True,"causality":True,"finite_gradients":True,"optimizer_updated":True})
    spec=json.loads((root/"package.json").read_text());art=json.loads((root/"provenance/recorded_artifacts.json").read_text())
    expected=art["historical_hierarchy"]["final_checkpoint_not_included"];h=hashlib.sha256()
    with args.hierarchy_checkpoint.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    require(h.hexdigest()==expected,"Historical checkpoint hash mismatch")
    width=spec["models"]["hierarchy"]["width"]
    vocab=json.loads((root/"configs/qwen06b_matched_hierarchy_function_v1.json").read_text())["vocab_size"]
    config=json.loads((root/"configs/qwen06b_matched_hierarchy_function_v1.json").read_text())
    hierarchy_config=config["models"]["hierarchical_distribution"]
    with torch.device("meta"):
        original_shape=models.HierarchicalDistributionStudent(vocab,width,config["model_max_context"],hierarchy_config["segment_length"],hierarchy_config["document_segments"])
    payload=torch.load(args.hierarchy_checkpoint,map_location="cpu",weights_only=True,mmap=True)
    require(isinstance(payload,dict) and "model" in payload and "result" in payload and "model_config" in payload,"Unknown historical checkpoint container")
    checkpoint=payload["model"]
    require(payload["result"]["stored_parameters"]==spec["models"]["hierarchy"]["stored_parameters"],"Historical checkpoint metadata count drift")
    declared=original_shape.state_dict()
    require(set(checkpoint)==set(declared),"Historical checkpoint key mismatch")
    require(all(checkpoint[k].shape==declared[k].shape for k in checkpoint),"Historical checkpoint tensor shape mismatch")
    require(sum(p.numel() for p in original_shape.parameters())==spec["models"]["hierarchy"]["stored_parameters"],"Historical stored parameter count mismatch")
    print("HISTORICAL_WORKFLOW_RECEIPT "+json.dumps({"status":"PASS_SYNTHETIC_AND_CHECKPOINT_LAYOUT_ONLY","identity_level":identity["historical_hierarchy"]["identity_level"],"not_byte_exact_deployed_source":True,"source_hashes":sources,"seeds":rows,"checkpoint_sha256":expected,"checkpoint_keys":len(checkpoint),"historical_parameters":spec["models"]["hierarchy"]["stored_parameters"],"dataset_content_read":False,"official_test_accessed":False,"is_language_quality_result":False,"threads":1,"device":"CPU","torch":torch.__version__},sort_keys=True),flush=True)
if __name__=="__main__":main()
