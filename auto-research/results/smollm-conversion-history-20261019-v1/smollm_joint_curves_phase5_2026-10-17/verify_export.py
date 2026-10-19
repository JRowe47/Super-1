"""Exercise the public five-mode loader and ordinary Transformers serialization."""
import gc
import tempfile

import torch
from transformers import AutoModelForCausalLM

from common import OUT, data, dump, setup
from load_candidate import load_candidate


@torch.no_grad()
def main():
    setup()
    model = load_candidate('lifted_dense', steps=3, native_export=True)
    ids = data()['audit'][:1, :32]
    expected = model(ids, use_cache=False).logits.clone()
    count = len(model.model.layers)
    values = sum(p.numel() for p in model.parameters())
    assert count == model.config.num_hidden_layers == 28
    with tempfile.TemporaryDirectory(prefix='phase5-native-export-') as directory:
        model.save_pretrained(directory, safe_serialization=True)
        del model
        gc.collect()
        reloaded = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation='eager').eval()
        actual = reloaded(ids, use_cache=False).logits
        difference = float((expected - actual).abs().max())
        assert difference == 0
        assert len(reloaded.model.layers) == reloaded.config.num_hidden_layers == count
    result = {
        'candidate': 'lifted_dense_n3',
        'public_loader': "load_candidate('lifted_dense', steps=3, native_export=True)",
        'format': 'Ordinary Transformers save_pretrained with safetensors',
        'reload': 'AutoModelForCausalLM.from_pretrained, local files, eager attention, FP32',
        'decoder_blocks': count,
        'unique_parameter_values': values,
        'max_reloaded_logit_abs_difference': difference,
        'input_tokens': 32,
        'passed': True,
        'temporary_export_removed': True,
    }
    dump(OUT / 'native_serialization.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
