"""Verify original decoder execution with parameters reconstructed from full curves."""
import argparse,json,pathlib,time,numpy as np,torch
from scipy import fft
from transformers import GPTNeoXForCausalLM
from insitu import *
torch.set_num_threads(3);torch.set_num_interop_threads(1)

@torch.inference_mode()
def main(checkpoints):
    begin=time.time();rng=np.random.default_rng(20261017)
    ids=torch.tensor(np.stack([np.r_[0,rng.choice(np.arange(100,50000),64,replace=False)] for _ in range(8)]))
    np.save(OUT/'baseline_input_ids.npy',ids.numpy());results=[]
    for step in [1000,143000]:
        model=GPTNeoXForCausalLM.from_pretrained(str(checkpoints/f'step{step}'),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
        original={f'{l}.{name}':p.detach().numpy().copy() for l,block in enumerate(model.gpt_neox.layers) for name,p in block.named_parameters()}
        reference=model(ids,use_cache=False).logits
        for basis,precision in [('fourier',32),('haar',32),('fourier',64),('haar',64)]:
            maxweight=0.
            for l,block in enumerate(model.gpt_neox.layers):
                for name,p in block.named_parameters():
                    x=original[f'{l}.{name}'].astype(np.float64).ravel()
                    if basis=='fourier':
                        c=fft.rfft(x,norm='ortho').astype(np.complex64 if precision==32 else np.complex128)
                        rec=fft.irfft(c.astype(np.complex128),n=len(x),norm='ortho')
                    else:
                        c,layout=haar(x);rec=ihaar(c.astype(np.float32 if precision==32 else np.float64),layout)
                    rec=rec.astype(np.float32).reshape(p.shape)
                    maxweight=max(maxweight,float(np.max(np.abs(rec-original[f'{l}.{name}']))))
                    p.copy_(torch.from_numpy(rec))
            logits=model(ids,use_cache=False).logits;d=logits-reference
            row={'step':step,'basis':basis,'coefficient_precision':precision,'all_decoder_parameters_reconstructed':True,'positions':ids.numel(),
              'max_weight_error':maxweight,'max_abs_logit_error':float(d.abs().max()),
              'relative_rms_logit_error':float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(reference)),
              'argmax_agreement':float((logits.argmax(-1)==reference.argmax(-1)).float().mean())}
            assert row['relative_rms_logit_error']<1e-5
            if precision==64: assert row['argmax_agreement']==1.
            results.append(row);print(row,flush=True)
        del model
    dump('functional_baseline.json',{'checks':results,'seconds':time.time()-begin,'training_steps':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=pathlib.Path,default=ROOT.parent/'pythia70m_spectral_2026-10-17'/'checkpoints')
    main(ap.parse_args().checkpoints)
