"""Validated four-step runtime entry point; experimental depths use profiles.load_profile."""
from . import load_router as _load_router

def load_router(model, use_case, tokenizer, context, **kwargs):
    if getattr(model, 'depth', None) != 4 or getattr(model, 'execution_depth', 4) != 4:
        raise ValueError('Four-step references cannot be installed in a different-depth model; use load_profile')
    return _load_router(model,use_case,tokenizer,context,**kwargs)
