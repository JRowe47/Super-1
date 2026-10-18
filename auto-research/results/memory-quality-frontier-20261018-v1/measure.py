"""One fresh CPU process per model/context; report total RSS separately from tensor bytes."""
import argparse,json,platform,resource,time
import torch
import run
from model import build,tensor_bytes
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--context',type=int,required=True);p.add_argument('--output',type=run.Path,required=True);a=p.parse_args();run.setup();C=run.C
assert a.context in C['memory_contexts'];out=run.H/'execution'
peak=lambda:resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
startup_peak=peak();m=build(C,a.arm,a.seed);path=out/f'{a.arm}-{a.seed}-inference.pt';state=torch.load(path,weights_only=True);m.load_state_dict(state['model']);del state;m.eval()
model_bytes=sum(p.numel()*p.element_size() for p in m.parameters());loaded_peak=peak();tokens=torch.tensor(list((run.H/'corpus.txt').read_bytes()[:a.context+C['decode_steps']]),dtype=torch.long)[None,:]
cache=None;start=time.perf_counter()
with torch.inference_mode():
    for offset in range(0,a.context,C['memory_chunk']):
        logits,cache=m(tokens[:,offset:min(offset+C['memory_chunk'],a.context)],cache)
        if time.perf_counter()-start>C['max_prefill_seconds']:raise TimeoutError('Prefill budget exhausted')
    prefill_seconds=time.perf_counter()-start;prefill_peak=peak();prefill_state_bytes=tensor_bytes(cache)
    decode_start=time.perf_counter()
    for i in range(C['decode_steps']):logits,cache=m(tokens[:,a.context+i:a.context+i+1],cache)
    decode_seconds=time.perf_counter()-decode_start;final_peak=peak();state_bytes=tensor_bytes(cache)
    assert torch.isfinite(logits).all()
record={'arm':a.arm,'seed':a.seed,'context':a.context,'checkpoint_sha256':run.sha(path),'model_bytes':model_bytes,'prefill_state_bytes':prefill_state_bytes,'decode_state_bytes':state_bytes,'retained_model_plus_state_bytes':model_bytes+state_bytes,'process_peak_rss_bytes':final_peak,'startup_peak_rss_bytes':startup_peak,'loaded_peak_rss_bytes':loaded_peak,'prefill_peak_rss_bytes':prefill_peak,'prefill_seconds':prefill_seconds,'decode_ms_per_token':decode_seconds/C['decode_steps']*1000,'decode_tokens':C['decode_steps'],'last_logits_sha256':run.hashlib.sha256(logits.numpy().tobytes()).hexdigest(),'scope':'Fresh process CPU FP32, batch1, streaming chunk64 prefill and teacher-forced cached decode. RSS includes runtime/allocator/transients; retained tensors are not peak total memory. Quality beyond256 untested.'}
run.save(a.output,record);print(json.dumps(record))
