"""Public loader, ordinary safetensors export and independent reload."""
import gc,tempfile
import torch
from transformers import AutoModelForCausalLM
from common import OUT,dump,setup,data
from models import load_candidate

@torch.no_grad()
def main():
    setup();results={};ids=data()['audit'][:1,:32]
    for kind in ['interpolated','retained']:
        model=load_candidate(kind,as_curve=False,steps=3)
        assert model.config.num_hidden_layers==len(model.model.layers)==28
        expected=model(ids,use_cache=False).logits
        with tempfile.TemporaryDirectory(prefix='phase6-export-') as directory:
            model.save_pretrained(directory,safe_serialization=True);del model;gc.collect()
            reloaded=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,
                torch_dtype=torch.float32,attn_implementation='eager').eval()
            difference=float((reloaded(ids,use_cache=False).logits-expected).abs().max());assert difference==0
            assert len(reloaded.model.layers)==reloaded.config.num_hidden_layers==28
        del reloaded;gc.collect()
        curve=load_candidate(kind,as_curve=True,steps=3)
        curve_diff=float((curve(ids,use_cache=False).logits-expected).abs().max());assert curve_diff<.001
        del curve;gc.collect()
        results[kind]={'saved_model_reload_max_logit_difference':difference,
            'public_curve_loader_max_logit_difference':curve_diff,'decoder_blocks':28,'input_tokens':32,
            'passed':True,'format':'Ordinary Transformers safetensors save_pretrained and from_pretrained'}
        print('PUBLIC_LOADER_CHECK',kind,results[kind],flush=True)
    dump(OUT/'native_serialization.json',results)

if __name__=='__main__':main()
