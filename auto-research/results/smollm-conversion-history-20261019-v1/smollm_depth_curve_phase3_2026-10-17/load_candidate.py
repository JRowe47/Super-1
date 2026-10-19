"""Load a phase-3 student into the pinned reference model's first block."""
import argparse, copy, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from depth_curve import make_from_state, install_student
from phase2_loader import load_first_block
from prepare_data import model_dir

ROOT=Path(__file__).resolve().parent


def restore(model,student_id,steps,shared_qk=False,materialized=False):
    info=json.loads((ROOT/'results'/f'training_{student_id}.json').read_text())
    state=load_file(str(ROOT/'checkpoints'/f'{student_id}.safetensors'))
    student=make_from_state(info['configuration'],state)
    if shared_qk:
        load_first_block(model,ROOT/'reference/joint_activation_svd_0.125.safetensors',
                         {'id':'joint_activation_svd_0.125','method':'joint_activation_svd','budget':.125})
    source=copy.deepcopy(model.model.layers[0])
    install_student(model,source,student,steps,materialized=materialized)
    return model.eval()


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('student',choices=['direct_192','direct_768','curve_fixed8','curve_mixed'])
    ap.add_argument('--steps',type=int,default=8)
    ap.add_argument('--shared-qk',action='store_true');ap.add_argument('--materialized',action='store_true')
    ap.add_argument('--prompt',default='The purpose of a scientific experiment is')
    args=ap.parse_args();torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(model_dir(),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    steps=1 if args.student.startswith('direct') else args.steps
    restore(model,args.student,steps,args.shared_qk,args.materialized)
    tok=AutoTokenizer.from_pretrained(model_dir(),local_files_only=True);inputs=tok(args.prompt,return_tensors='pt')
    out=model.generate(**inputs,max_new_tokens=40,do_sample=False,pad_token_id=tok.eos_token_id)
    print(tok.decode(out[0],skip_special_tokens=True))


if __name__=='__main__':main()
