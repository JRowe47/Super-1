import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import copy,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
import core as C

torch.set_num_threads(1);torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parent

def verify():
    begin=time.process_time();results={}
    for dtype,tol in [(torch.float32,2e-6),(torch.float64,2e-12)]:
        name=str(dtype);s=C.init(18719,dtype);w=C.decode(s)
        gen=torch.Generator().manual_seed(52871)
        x=torch.randn(9,C.D,generator=gen,dtype=dtype);y=torch.randint(0,2,(9,C.R),generator=gen).to(dtype)
        checks={}
        checks['roundtrip']=max(float((C.dec(C.enc(v)).reshape(v.shape)-v).abs().max()) for v in w.values())
        checks['isometry']=max(abs(float(v.norm()-s[k].norm())) for k,v in w.items())
        checks['forward']=float((C.forward(s,x)-C.ports(w,x)[0][-1]).abs().max())
        g,acts,gates,_=C.dense_local(w,x,y);ref=C.dense_reference(w,x,y)
        assert all(bool(g.any()) and bool((~g).any()) for g in gates)
        checks['local_reference']=max(float((g[k]-ref[k]).abs().max()) for k in g)
        curve_new=C.apply(s,C.encode(g),C.SHAPES);dense_new=C.decode(curve_new)
        direct={k:w[k]-C.LR*(ref[k]+(C.WD_WB if k.startswith(('W','B')) else C.WD_VU)*w[k]) for k in w}
        checks['dense_curve_local_step']=max(float((dense_new[k]-direct[k]).abs().max()) for k in w)
        dw={k:v.detach().clone().requires_grad_(k in C.WK) for k,v in w.items()}
        dx=x.clone().requires_grad_();dense_loss=F.binary_cross_entropy_with_logits(C.ports(dw,dx)[0][-1],y)
        dg=torch.autograd.grad(dense_loss,[dw[k] for k in C.WK]+[dx])
        cs={k:v.detach().clone().requires_grad_(k in C.WK) for k,v in s.items()}
        cx=x.clone().requires_grad_();curve_loss=C.loss_value(cs,cx,y)
        cg=torch.autograd.grad(curve_loss,[cs[k] for k in C.WK]+[cx])
        checks['exact_gradient']=max(float((C.enc(a)-b).abs().max()) for a,b in zip(dg[:-1],cg[:-1]))
        checks['input_gradient']=float((dg[-1]-cg[-1]).abs().max())
        # Confirm that applying the coordinatewise rule to the coefficient layout is a different operation.
        fake_w={k:v.reshape(C.SHAPES[k]) for k,v in s.items()}
        wrong,*_=C.dense_local(fake_w,x,y)
        wrong_gap=max(float((wrong[k].flatten()-C.enc(g[k])).abs().max()) for k in C.WK)
        assert wrong_gap>100*tol
        # Pair updates share the drive exactly; only initial mismatch is decayed.
        checks['paired_decay']=max(float(((dense_new['V'+str(j)]-dense_new['U'+str(j-1)].T)-
                                        (1-C.LR*C.WD_VU)*(w['V'+str(j)]-w['U'+str(j-1)].T)).abs().max()) for j in (1,2))
        # Storage isolation includes all forward and feedback matrices, with nonzero values.
        a=C.clone(s);b=C.clone(s);a['V1'].add_(1)
        assert torch.equal(b['V1'],s['V1']) and not torch.equal(a['V1'],s['V1'])
        assert all(value<=tol for value in checks.values()), (name,checks,tol)
        # Local calls must remain possible when autograd.grad is forbidden.
        old=torch.autograd.grad
        def forbidden(*args,**kwargs): raise AssertionError('Autograd called by local path')
        try:
            torch.autograd.grad=forbidden
            for arm in ('random_fixed','warm_frozen','adaptive'): C.update(s,x,y,arm)
        finally: torch.autograd.grad=old
        results[name]={'tolerance':tol,'checks':checks,'wrong_coordinate_rule_gap':wrong_gap}
    d=C.make_data(94957,8,32,16)
    assert all(int(i)<16*t for t,ix in enumerate(d['replay_ids']) for i in ix if i>=0)
    assert int(d['stale_replayed_facts'].sum())>0
    assert not bool(d['stale_replayed_facts'][96:].any())
    result={'status':'PASS','numeric':results,'causal_replay':True,'stale_labels_preserved':True,
            'arm_isolation':True,'local_update_without_autograd':True,'cpu_seconds':time.process_time()-begin}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': verify()
