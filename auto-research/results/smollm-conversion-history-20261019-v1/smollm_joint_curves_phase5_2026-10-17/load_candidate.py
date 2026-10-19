"""Load function weights, or compile a fixed-depth native transformer."""
from common import ROOT,load_model
from joint_model import make_banks,load_banks,set_model,export_native,project_dense,lift_dense


def load_candidate(kind='orthogonal_curve',steps=3,native_export=False):
    if kind not in ['raw_curve','orthogonal_curve','dense','projected_dense','lifted_dense'] or steps not in [2,3,4]:
        raise ValueError('Unsupported candidate')
    if kind=='dense' and steps==4:raise ValueError('Dense controls were trained/evaluated only at N=3 or N=2')
    model=load_model();original=list(model.model.layers)
    if kind in ['projected_dense','lifted_dense']:
        dense=make_banks(original,'dense');load_banks(dense,ROOT/'checkpoints/dense.safetensors')
        banks,_=(project_dense if kind=='projected_dense' else lift_dense)(original,dense)
    else:
        banks=make_banks(original,kind);load_banks(banks,ROOT/'checkpoints'/f'{kind}.safetensors')
    banks.requires_grad_(False)
    if native_export:
        export_native(model,original,banks,steps)
        model.config.num_hidden_layers=len(model.model.layers)
        # The returned model contains only native decoder blocks and needs no
        # external bank at inference. Sampling/folding happens once above.
        return model
    set_model(model,original,banks,steps)
    object.__setattr__(model,'phase5_original_blocks',original)
    object.__setattr__(model,'phase5_banks',banks)
    return model
