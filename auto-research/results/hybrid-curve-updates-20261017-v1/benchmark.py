"""Post-main fixed-checkpoint timing diagnostic; no retraining or selection."""
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from experiment import ROOT,Model,decode,save,sha

torch.set_num_threads(1);torch.set_num_interop_threads(1)
checkpoint=ROOT/'main/s101721/backprop.pt'
state=torch.load(checkpoint,weights_only=True);model=Model();model.load_state_dict(state['model'])
a=decode(model.c1).detach().reshape(32,17).requires_grad_()
b=decode(model.c2).detach().reshape(8,33).requires_grad_()
rows=[]
for size in [1,64,256]:
    torch.manual_seed(811+size);x=torch.randn(size,16);y=torch.randint(0,8,(size,))
    def curve():
        z=model(x)[0];g=torch.autograd.grad(F.cross_entropy(z,y),(model.c1,model.c2))
        return z,[decode(c-.5*d) for c,d in zip([model.c1,model.c2],g)]
    def dense():
        z=F.linear(F.linear(x,a[:,:-1],a[:,-1]).tanh(),b[:,:-1],b[:,-1])
        g=torch.autograd.grad(F.cross_entropy(z,y),(a,b))
        return z,[(w-.5*d).reshape(-1) for w,d in zip([a,b],g)]
    # Timing proposal is the same numerical operation at every repetition.
    # Parameters remain at the frozen checkpoint to avoid trajectory effects.
    cz,cw=curve();dz,dw=dense()
    error=max(float((cz-dz).abs().max().detach()),max(float((c-d).abs().max().detach()) for c,d in zip(cw,dw)))
    assert error<=2e-6,('fixed_checkpoint_parity',size,error)
    times={'curve':[],'dense':[]}
    for _ in range(10):curve();dense()
    for repeat in range(200):
        for name in (['curve','dense'] if repeat%2 else ['dense','curve']):
            t=time.process_time_ns();(curve if name=='curve' else dense)();times[name].append((time.process_time_ns()-t)/1e6)
    rows.append(dict(batch=size,maximum_parity_error=error,
        cpu_ms_median={k:float(np.median(v)) for k,v in times.items()},
        cpu_ms_p10={k:float(np.quantile(v,.1)) for k,v in times.items()},
        cpu_ms_p90={k:float(np.quantile(v,.9)) for k,v in times.items()},samples=times))
save(ROOT/'COMPUTE_DIAGNOSTIC.json',dict(status='COMPLETE',registered_after_main=True,
    checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),no_retraining=True,
    protocol='10 warmups, 200 interleaved pairs; process CPU; fixed parameters; forward/backward and proposed SGD update; full inverse reconstruction in curve result is included.',rows=rows))
print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in rows],indent=2))
