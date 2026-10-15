
import copy,hashlib,importlib.util,json,pathlib,sys,types
import torch
from torch.nn import functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
root=pathlib.Path('/Volumes/SP AI 01_16/repro-packages/official-nanogpt-vs-proven-hierarchy-10m-20260815-v1r2-email-safe-attempt1')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
package=types.ModuleType("representation_experiments");package.__path__=[str(root/'source/historical_hierarchy/representation_experiments')];sys.modules[package.__name__]=package
load("representation_experiments.common",root/'source/historical_hierarchy/representation_experiments/common.py')
models=load("representation_experiments.predictive_models",root/'source/historical_hierarchy/representation_experiments/predictive_models.py')
geometry=types.ModuleType('geometry_candidate');exec(compile(GEOMETRY_SOURCE,'conditioned_fast_rotation.py','exec'),geometry.__dict__)
nano=load("geometry_pinned_nanogpt",root/'source/official_nanogpt/third_party/nanogpt/model.py')
def require(ok,message):
 if not ok:raise RuntimeError(message)
rows=[]
for seed in (73,191,307):
 torch.manual_seed(seed)
 baseline=models.HierarchicalDistributionStudent(64,16,272,32,8).double().cpu()
 x=torch.randint(64,(1,272));target=torch.randint(64,(1,272))
 with torch.no_grad():initial=baseline(x).logits.detach().clone()
 arms={"baseline":baseline,"oriented_cross":geometry.install_fast_rotation(copy.deepcopy(baseline),"oriented_cross"),"alignment_dot":geometry.install_fast_rotation(copy.deepcopy(baseline),"alignment_dot")}
 counts={k:sum(p.numel() for p in m.parameters()) for k,m in arms.items()}
 require(counts["oriented_cross"]==counts["alignment_dot"],"Query control parameter mismatch")
 for kind,model in arms.items():
  with torch.no_grad():
   identity=model(x).logits
   require(torch.equal(identity,initial),"Identity changed historical transition")
   if kind!="baseline":model.fast_cell.gain.fill_(.3)
   full=model(x);changed=x.clone();changed[:,128:]=(changed[:,128:]+1)%64
   require(torch.equal(full.logits[:,:128],model(changed).logits[:,:128]),"Geometry causal violation")
   state=None;values=[]
   for j in range(x.shape[1]):
    logits,state,_=model.step(x[:,j],state);values.append(logits)
   require(torch.allclose(full.logits,torch.stack(values,1),atol=1e-9,rtol=1e-9),"Geometry forward/stream mismatch")
  optimizer=torch.optim.AdamW(model.parameters(),lr=.001);before={n:p.detach().clone() for n,p in model.named_parameters()}
  loss=F.cross_entropy(model(x).logits.reshape(-1,64),target.reshape(-1));loss.backward()
  require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"Nonfinite/missing gradient")
  if kind!="baseline":require(bool(model.fast_cell.gain.grad.abs().sum()>0),"Disconnected geometry")
  optimizer.step()
  require(any(not torch.equal(before[n],p) for n,p in model.named_parameters()),"No optimizer update")
  rows.append({"seed":seed,"arm":kind,"parameters":counts[kind],"identity_exact":True,"causality":True,"streaming_match":True,"finite_gradients":True,"optimizer_updated":True})
 cfg=nano.GPTConfig(block_size=272,vocab_size=64,n_layer=1,n_head=2,n_embd=16,dropout=0,bias=False)
 model=nano.GPT(cfg).double().cpu();logits,loss=model(x,target);loss.backward()
 require(bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"nanoGPT synthetic failed")
 rows.append({"seed":seed,"arm":"nanoGPT_workflow_control","parameters":sum(p.numel() for p in model.parameters()),"finite_loss_and_gradients":True,"not_parameter_matched":True})
receipt={"status":"PASS_NEW_GEOMETRY_SYNTHETIC_ONLY","model_identity":"historical compatible HierarchicalDistributionStudent snapshot, not byte-exact original source","historical_clock_fixture":{"segment_length":32,"document_segments":8,"tokens":272},"new_geometry":"native fast-GRU output conditioned pair rotation","controls":"unmodified historical baseline and equal-count alignment-dot query; nanoGPT is a workflow control, not matched quality comparison","scope":"No language benchmark, no new task-data reads, no SOTA or novelty claim","rows":rows,"threads":1,"device":"CPU","source_sha256":{"geometry_wrapper":hashlib.sha256(GEOMETRY_SOURCE.encode()).hexdigest(),"historical_model":hashlib.sha256((root/'source/historical_hierarchy/representation_experiments/predictive_models.py').read_bytes()).hexdigest(),"nanogpt":hashlib.sha256((root/'source/official_nanogpt/third_party/nanogpt/model.py').read_bytes()).hexdigest()}}
print('GEOMETRY_RECEIPT '+json.dumps(receipt,sort_keys=True),flush=True)
