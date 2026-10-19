"""Recreate a phase-4 model from original model files plus saved slopes."""
from pathlib import Path
from common import ROOT,load_model
from curve_model import HermiteBank,install
from train import load_bank


def load_candidate(resolutions=None,materialized=False):
    """Example resolutions={4:3,12:2}; omitted windows remain native.

    Resolution is fixed for an entire sequence. Start a fresh DynamicCache when
    changing it. Returns model plus banks, keeping the bank references alive.
    """
    resolutions={4:3,12:3} if resolutions is None else resolutions
    model=load_model();original=list(model.model.layers);banks={};specs={}
    for start,steps in resolutions.items():
        if start not in [4,12] or steps not in [2,3,4]:raise ValueError('Unsupported window/resolution')
        bank=HermiteBank(original[start:start+4]);load_bank(bank,ROOT/'checkpoints'/f'window_{start}.safetensors')
        bank.requires_grad_(False);banks[start]=bank;specs[start]={'bank':bank,'steps':steps}
    install(model,original,specs,materialized=materialized)
    # Keep native originals available: every frozen knot must remain alive, and
    # omitted parameters are intentionally retained, not compressed away.
    object.__setattr__(model,'phase4_original_blocks',original)
    object.__setattr__(model,'phase4_banks',banks)
    return model,banks
