"""Run the independent dense arm while the sequential trainer handles curves.

The sequential train.py skips this arm once its completion record exists. Both
workers use four CPU threads, separate model/optimizer state, the same recorded
batches, and separate checkpoint/report paths. Do not launch duplicate dense
workers. Prefill timing is performed only after all training workers exit.
"""
import json,time
import numpy as np
import torch
from common import ROOT,OUT,dump,setup,load_model,hashes,data
from train import train_arm
from curve_model import restore

if __name__=='__main__':
    setup();p=json.loads((OUT/'protocol.json').read_text());windows=data();model=load_model();original=list(model.model.layers);before=hashes(model)
    trainref=torch.load(ROOT/'cache/train.pt',weights_only=True);valref=torch.load(ROOT/'cache/validation.pt',weights_only=True)
    indices=torch.tensor(np.load(ROOT/'data/training_indices.npy'),dtype=torch.long)
    train_arm(model,original,'dense',windows,trainref,valref,indices,p['peak_lrs']['dense'])
    restore(model,original);assert before==hashes(model)
    dump(OUT/'dense_parallel_integrity.json',{'original_hashes_unchanged':True,'threads':4,
        'concurrency':'Overlapped with curve training in another four-thread process. Training wall time is descriptive and not a controlled speed comparison; inference timing runs after all workers finish.'})
