"""Restore one saved block-0 candidate without recomputing SVD or calibration.

Requires the pinned FP32 CPU/eager SmolLM2 model and the experiment's selection
metadata. The remaining 29 blocks are retained from that reference model.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from prepare_data import model_dir
from representations import (FactorCurveLinear, JointQKCurve, JointQKView,
                             GroupedCurveMLP, wrap_unchanged, rank_for_budget)

ROOT = Path(__file__).resolve().parent


def load_first_block(model, checkpoint, spec):
    """Load a saved candidate into a fresh reference model; return that model."""
    cfg = model.config
    expected = (576, 1536, 30, 9, 3)
    observed = (cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers,
                cfg.num_attention_heads, cfg.num_key_value_heads)
    if observed != expected or getattr(cfg, 'pretraining_tp', 1) != 1:
        raise ValueError('This checkpoint loader is specific to SmolLM2-135M.')
    if next(model.parameters()).device.type != 'cpu' or next(model.parameters()).dtype != torch.float32:
        raise ValueError('Reproduce this experiment with a CPU FP32 reference model.')
    if getattr(cfg, '_attn_implementation', None) != 'eager':
        raise ValueError('The measured reference uses eager attention.')
    state = load_file(str(checkpoint), device='cpu')
    block = copy.deepcopy(model.model.layers[0])
    method = spec['method']
    if method == 'activation_svd':
        for name in ['q_proj', 'k_proj']:
            weight = getattr(block.self_attn, name).weight
            rank = rank_for_budget(weight, spec['budget'])
            m, n = weight.shape
            module = FactorCurveLinear(torch.empty(m, rank), torch.empty(rank, n))
            setattr(block.self_attn, name, module)
    elif method == 'joint_activation_svd':
        weight = torch.cat([block.self_attn.q_proj.weight, block.self_attn.k_proj.weight])
        rank = rank_for_budget(weight, spec['budget'])
        bank = JointQKCurve(torch.empty(768, rank), torch.empty(rank, 576), 576)
        block.add_module('joint_qk_curve', bank)
        block.self_attn.q_proj = JointQKView(bank, 'q')
        block.self_attn.k_proj = JointQKView(bank, 'k')
    elif method == 'mlp_groups':
        indices = state['mlp.selected_neurons']
        if len(indices) != spec['groups'] * 96:
            raise ValueError('Checkpoint and group specification disagree.')
        block.mlp = GroupedCurveMLP(block.mlp, indices)
    else:
        raise ValueError(f'No saved checkpoint is provided for method {method}.')
    wrap_unchanged(block)
    block.load_state_dict(state, strict=True)
    model.model.layers[0] = block.eval()
    return model.eval()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('candidate_id', help='An id with a checkpoint in checkpoints/')
    ap.add_argument('--prompt', default='The purpose of a scientific experiment is')
    args = ap.parse_args()
    torch.set_num_threads(4)
    selected = json.loads((ROOT/'results/selection.json').read_text())['selected']
    spec = next((s for s in selected if s['id'] == args.candidate_id), None)
    if spec is None:
        raise ValueError(f'Unknown candidate: {args.candidate_id}')
    model = AutoModelForCausalLM.from_pretrained(model_dir(), local_files_only=True,
                    torch_dtype=torch.float32, attn_implementation='eager').eval()
    load_first_block(model, ROOT/'checkpoints'/f'{args.candidate_id}.safetensors', spec)
    tok = AutoTokenizer.from_pretrained(model_dir(), local_files_only=True)
    inputs = tok(args.prompt, return_tensors='pt')
    output = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                            pad_token_id=tok.eos_token_id)
    print(tok.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
