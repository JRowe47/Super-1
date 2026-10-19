"""Exploratory validation-only norm control; never selects trained weights."""
import gc,math
import torch
from common import ROOT,OUT,dump,setup,load_model,data,hashes
from models import candidate,install,restore,TIMES
from evaluation import evaluate

@torch.no_grad()
def main():
    setup()
    dump(OUT/'protocol_addendum_norm.json',{
        'status':'Exploratory diagnostic added after the 300-update validation checkpoints, before any phase-6 fresh-audit evaluation.',
        'question':'Does sampled matrix-norm attenuation help explain sensitivity to residual strength?',
        'rule':'For each of the seven 2-D projection matrices, rescale the interpolated sample to sqrt((1-u)*||W_j||_F^2 + u*||W_(j+1)||_F^2), using adjacent ORIGINAL knots. Leave both RMSNorm vectors unchanged.',
        'residual_scales':[1.,4/3],'data':'All 32 validation windows only.',
        'training_changes':'None. No candidate selection, training, LR changes or fresh-audit comparison uses these controls.'})
    windows=data();reference=torch.load(ROOT/'cache/validation.pt',weights_only=True)
    model=load_model();original=list(model.model.layers);before=hashes(model);rows=[]
    for scale in [1.,4/3]:
        blocks=candidate(original,'interpolated');factors={}
        for s in ['4','12']:
            for i,t in enumerate(TIMES):
                j=math.floor(t);u=t-j;left=dict(original[int(s)+j].named_parameters());right=dict(original[int(s)+j+1].named_parameters())
                for name,p in blocks[s][i].named_parameters():
                    if p.ndim!=2:continue
                    desired=((1-u)*left[name].double().square().sum()+u*right[name].double().square().sum()).sqrt()
                    factor=float(desired/p.double().norm());p.mul_(factor)
                    factors[f'{s}.{i}.{name}']=factor
                blocks[s][i].self_attn.o_proj.weight.mul_(scale)
                blocks[s][i].mlp.down_proj.weight.mul_(scale)
        install(model,original,blocks);metrics=evaluate(model,windows['validation'],reference)
        rows.append({'id':f'matrix_norm_restored_scale_{scale:.6f}','residual_scale':scale,
            'normalization_factors':factors,'metrics':metrics,'threads':torch.get_num_threads()})
        dump(OUT/'exploratory_norm_control.json',rows)
        print('NORM_CONTROL',scale,metrics['perplexity_ratio_to_original'],metrics['mean_kl'],flush=True)
        restore(model,original);del blocks;gc.collect()
    assert before==hashes(model)

if __name__=='__main__':main()
