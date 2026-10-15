
"""Synthetic workflow acceptance check; not a language benchmark or SOTA result."""
import argparse,copy,hashlib,importlib.util,json,os,pathlib,sys,time
def require(ok,message):
    if not ok: raise RuntimeError(message)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,"Missing module: "+str(path))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--hierarchy-root",type=pathlib.Path,required=True)
    p.add_argument("--hierarchy-protocol",type=pathlib.Path,required=True)
    p.add_argument("--nanogpt-model",type=pathlib.Path,required=True)
    p.add_argument("--nanogpt-provenance",type=pathlib.Path,required=True)
    p.add_argument("--confirm-synthetic-only",action="store_true",required=True)
    args=p.parse_args()
    started=time.monotonic();pins=json.loads(args.hierarchy_protocol.read_text())["source_hashes"]
    common=pathlib.Path(os.path.commonpath(list(pins)))
    checked=[]
    for old,expected in pins.items():
        path=args.hierarchy_root/pathlib.Path(old).relative_to(common)
        require(sha(path)==expected,"Hierarchy source drift: "+str(path));checked.append({"path":str(path),"sha256":expected})
    provenance=json.loads(args.nanogpt_provenance.read_text())["official_nanogpt"]
    require(sha(args.nanogpt_model)==provenance["vendored_model_sha256"],"nanoGPT source drift")
    checked.append({"path":str(args.nanogpt_model),"sha256":sha(args.nanogpt_model)})
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rotation=load_module("workflow_regularized_reference",args.hierarchy_root/"regularized_rotation.py")
    reference=rotation.reference.reference.reference
    nano=load_module("workflow_pinned_nanogpt",args.nanogpt_model)
    seed_rows=[]
    # Fixed small acceptance-test shapes, not experimental hyperparameter tuning.
    for seed in (73,191,307):
        torch.manual_seed(seed)
        base=reference.PersistentMultiScaleLM(256,32,4).float().cpu()
        x=torch.randint(256,(2,16));target=torch.randint(256,(2,16))
        checks={}
        with torch.no_grad(): identity_reference=base(x).detach().clone()
        for arm,model in [("original",base),("cross_scale",rotation.RegularizedRotation(copy.deepcopy(base),True))]:
            if arm=="cross_scale":
                with torch.no_grad():
                    identity=model(x);reference_logits=identity_reference
                require(torch.allclose(identity,reference_logits,atol=1e-6,rtol=1e-5),"Identity-centered geometry changed reference")
                checks["identity_max_abs"]=float((identity-reference_logits).abs().max())
                with torch.no_grad():model.rotation_gain.fill_(.2)
            model.eval()
            with torch.no_grad():
                full,state=model(x,return_state=True)
                left,carry=model(x[:,:8],return_state=True)
                right,split_state=model(x[:,8:],state=carry,return_state=True)
                joined=torch.cat((left,right),1)
                changed=x.clone();changed[:,8:]=(changed[:,8:]+1)%256
                altered=model(changed)
            require(torch.allclose(full,joined,atol=1e-6,rtol=1e-5),arm+" split-stream logits drift")
            require(torch.allclose(state,split_state,atol=1e-6,rtol=1e-5),arm+" split-stream state drift")
            require(torch.equal(full[:,:8],altered[:,:8]),arm+" causal prefix violation")
            require(tuple(full.shape)==(2,16,256),arm+" output shape")
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            before={name:value.detach().clone() for name,value in model.named_parameters()}
            loss=F.cross_entropy(model(x).reshape(-1,256),target.reshape(-1))
            require(bool(torch.isfinite(loss)),arm+" nonfinite loss");loss.backward()
            grads={name:value.grad for name,value in model.named_parameters()}
            require(all(g is None or bool(torch.isfinite(g).all()) for g in grads.values()),arm+" nonfinite gradient")
            require(any(g is not None and bool(g.abs().sum()>0) for g in grads.values()),arm+" no gradient")
            if arm=="cross_scale":
                require(bool(model.rotation_gain.grad.abs().sum()>0),"Disconnected transport gradient")
            optimizer.step()
            require(all(bool(torch.isfinite(value).all()) for value in model.parameters()),arm+" nonfinite updated parameters")
            require(any(not torch.equal(before[name],value) for name,value in model.named_parameters()),arm+" optimizer did not update")
            checks[arm]={"parameters":sum(p.numel() for p in model.parameters()),"synthetic_loss":float(loss.detach()),"split_logit_max_abs":float((full-joined).abs().max()),"split_state_max_abs":float((state-split_state).abs().max()),"causality":True,"finite_gradients":True,"optimizer_updated":True}
        cfg=nano.GPTConfig(block_size=16,vocab_size=256,n_layer=2,n_head=4,n_embd=32,dropout=0,bias=False)
        model=nano.GPT(cfg).float().cpu().eval()
        logits,loss=model(x,target)
        require(tuple(logits.shape)==(2,16,256),"nanoGPT shape")
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        before=next(model.parameters()).detach().clone();loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"nanoGPT gradient finiteness")
        optimizer.step()
        require(not torch.equal(before,next(model.parameters())),"nanoGPT optimizer did not update")
        checks["nanogpt"]={"parameters":sum(p.numel() for p in model.parameters()),"finite_loss":bool(torch.isfinite(loss)),"finite_gradients":True,"optimizer_updated":True}
        seed_rows.append({"seed":seed,"checks":checks})
    print("WORKFLOW_RECEIPT "+json.dumps({"status":"PASS_SYNTHETIC_ACCEPTANCE_ONLY","synthetic_only":True,"is_language_benchmark":False,"is_parameter_matched_comparison":False,"official_test_accessed":False,"dataset_content_read":False,"cloud_resources_created":0,"device":"CPU","threads":1,"torch":torch.__version__,"python":sys.version,"source_hashes_checked":checked,"seeds":seed_rows,"elapsed_seconds":time.monotonic()-started},sort_keys=True),flush=True)
if __name__=="__main__":main()
