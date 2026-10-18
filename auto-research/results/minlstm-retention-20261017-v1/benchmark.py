"""Isolated actual CPU timing; optimized nn.LSTM is intentionally the baseline."""
import argparse, gc, json, platform, resource, time
import numpy as np
import torch
from experiment import ROOT,save
from model import Model, ARMS

def measure(fn,n,warmup):
    for _ in range(warmup):fn()
    cpu=[];wall=[];gc.disable()
    try:
        for _ in range(n):
            c=time.process_time_ns();w=time.perf_counter_ns();fn()
            wall.append((time.perf_counter_ns()-w)/1e6);cpu.append((time.process_time_ns()-c)/1e6)
    finally:gc.enable()
    return dict(repetitions=n,warmup=warmup,cpu_ms_median=float(np.median(cpu)),wall_ms_median=float(np.median(wall)),
                cpu_ms_p10=float(np.quantile(cpu,.1)),cpu_ms_p90=float(np.quantile(cpu,.9)),cpu_ms_samples=cpu)

def main():
    p=argparse.ArgumentParser();p.add_argument('--canary',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(882)
    rows=[]
    for arm in ARMS:
        model=Model(arm);opt=torch.optim.AdamW(model.parameters(),lr=.003,betas=(.9,.95),weight_decay=.01)
        for length in ([64] if a.canary else [64,256,1024]):
            x=torch.randint(0,52,(32,length));y=torch.randint(0,8,(32,length))
            def train():
                model.train();opt.zero_grad(set_to_none=True);z,_=model(x)
                loss=torch.nn.functional.cross_entropy(z.reshape(-1,8),y.reshape(-1));loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            rows.append(dict(arm=arm,operation='training_step',batch=32,length=length,
                             **measure(train,5 if a.canary else 20,3 if a.canary else 5)))
            if not a.canary:
                model.eval()
                with torch.inference_mode():
                    rows.append(dict(arm=arm,operation='prefill',batch=32,length=length,**measure(lambda:model(x),30,5)))
        if not a.canary:
            for batch in (1,32):
                x=torch.randint(0,52,(batch,1));state=None
                def step():
                    nonlocal state
                    _,state=model(x,state)
                model.eval()
                with torch.inference_mode():
                    rows.append(dict(arm=arm,operation='decode',batch=batch,length=1,**measure(step,200,20)))
    result=dict(status='COMPLETE',hardware=platform.processor(),torch=torch.__version__,threads=1,
                mkldnn_enabled=torch.backends.mkldnn.enabled,scan='Hillis-Steele, O(T log T) work',
                native_lstm='torch.nn.LSTM CPU optimized backend',rows=rows,unix=time.time())
    save(ROOT/('CANARY.json' if a.canary else 'COMPUTE.json'),result)
    print(json.dumps({r['arm']+':'+r['operation']+':'+str(r['batch'])+':'+str(r['length']):r['cpu_ms_median'] for r in rows},indent=2))
if __name__=='__main__':main()
