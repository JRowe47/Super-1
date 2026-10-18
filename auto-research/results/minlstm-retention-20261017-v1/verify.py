import copy, json, time
from pathlib import Path
import torch
from model import Model, ARMS
from tasks import episode, check_oracle

def verify():
    torch.set_num_threads(1)
    rows = []
    for arm in ARMS:
        for dtype in (torch.float64, torch.float32):
            torch.manual_seed(881)
            a=Model(arm).to(dtype); b=copy.deepcopy(a)
            # Activate gates after the controlled initialization, so tests
            # cannot pass only because gate weights started at zero.
            with torch.no_grad():
                for p in a.parameters(): p.add_(.03*torch.randn_like(p))
            b.load_state_dict(a.state_dict())
            x=torch.randn(3,17,32,dtype=dtype,requires_grad=True)
            y=x.detach().clone().requires_grad_()
            h=torch.randn(3,a.width,dtype=dtype)
            initial=(h,torch.randn_like(h)) if a.kind=='lstm' else h
            out,st=a.recurrent(x,initial); ref,rst=b.recurrent(y,initial,True)
            (out.square().mean()).backward();(ref.square().mean()).backward()
            error=float((out-ref).detach().abs().max())
            grad=max(float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()) if p.grad is not None)
            inp=float((x.grad-y.grad).abs().max())
            left,carry=a.recurrent(x[:,:7],initial)
            right,_=a.recurrent(x[:,7:],carry)
            chunk=float((out-torch.cat((left,right),1)).detach().abs().max())
            oa=torch.optim.AdamW(a.parameters(),lr=.003);ob=torch.optim.AdamW(b.parameters(),lr=.003)
            oa.step();ob.step()
            opt=max(float((p-q).detach().abs().max()) for p,q in zip(a.parameters(),b.parameters()))
            with torch.no_grad():
                updated=float((a.recurrent(x,initial)[0]-b.recurrent(y,initial,True)[0]).abs().max())
            tol=2e-12 if dtype==torch.float64 else 2e-6
            # Adam's 1e-8 denominator amplifies rounding in almost-zero gradients.
            # Keep tight output/gradient tests; additionally test the updated
            # function, and record the observed parameter-space roundoff.
            opt_tol=2e-12 if dtype==torch.float64 else 5e-5
            assert max(error,grad,inp,chunk)<tol,(arm,dtype,error,grad,inp,chunk)
            assert opt<opt_tol and updated<tol,(arm,dtype,opt,updated)
            rows.append(dict(arm=arm,dtype=str(dtype),output_max_abs=error,gradient_max_abs=grad,
                             input_gradient_max_abs=inp,chunk_max_abs=chunk,optimizer_max_abs=opt,
                             post_optimizer_output_max_abs=updated,tolerance=tol,optimizer_parameter_tolerance=opt_tol))
    # Literal gate formula and stable implementation agree away from underflow.
    f=torch.linspace(-25,25,101,dtype=torch.float64);i=f.flip(0)*.8
    direct=i.sigmoid()/(f.sigmoid()+i.sigmoid())
    stable=torch.sigmoid(torch.nn.functional.logsigmoid(i)-torch.nn.functional.logsigmoid(f))
    gate=float((direct-stable).abs().max());assert gate<1e-14
    for mode in ('retention','conflict','reopen','train'):
        x,y,_=episode(802,batch=7,delay=31,mode=mode);assert check_oracle(x,y)
    # Extreme decay does not require division by underflowed cumulative products.
    from model import affine_scan
    a=torch.full((2,1031,4),.01);b=torch.randn_like(a);h=torch.ones(2,4)
    parallel=affine_scan(a,b,h);seq=[]
    for at,bt in zip(a.unbind(1),b.unbind(1)):h=at*h+bt;seq.append(h)
    stress=float((parallel-torch.stack(seq,1)).abs().max());assert stress<1e-6
    result=dict(status='PASS',checks=rows,normalized_gate_max_abs=gate,
                long_underflow_stress_max_abs=stress,oracle='PASS',finished_unix=time.time())
    Path(__file__).with_name('VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':verify()
