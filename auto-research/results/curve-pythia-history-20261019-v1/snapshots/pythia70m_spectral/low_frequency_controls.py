"""Post-hoc random-order low-frequency controls for the observed PCA effect."""
import numpy as np, torch
from common import *
from behavior import load_model, text_inputs, text_score

@torch.inference_mode()
def main():
    dump('low_frequency_addendum.json',{'reason':'The primary PCA low-frequency result improved over native ordering; add three preexisting random orderings to measure ordering sensitivity.',
       'timing':'Specified after primary results, before these random-low results','new_optimizer_steps':0})
    model=load_model(8000); inputs=text_inputs(); full,teachers=text_score(model,inputs,keep_teacher=True)
    source=state(4000); orders=get_orderings(); initial={};delta={};params={}
    for l,block in enumerate(model.gpt_neox.layers):
        for fam,attr in [('mlp_in','dense_h_to_4h'),('mlp_out','dense_4h_to_h')]:
            k=(l,fam);params[k]=getattr(block.mlp,attr).weight
            initial[k]=source[f'gpt_neox.layers.{l}.mlp.{attr}.weight'].copy()
            delta[k]=params[k].numpy().copy()-initial[k]
    del source
    rows=json.loads((OUT/'sparse_interventions.json').read_text())
    reverted=next(r for r in rows if r['variant']=='reverted_mlp_weights')
    for fraction in [.1,.25]:
        for name in ['random0','random1','random2']:
            label='fourier_'+name+'_low'
            if any(r['variant']==label and r['fraction']==fraction for r in rows):continue
            kept=0;retained=0.
            for (l,fam),d in delta.items():
                p=orders[(l,name)];axis=axis_for(fam)
                c=real_fft(ordered(d,fam,p),axis)
                m,k=mask_coefficients(c,fraction,'low',axis=axis)
                upd=unordered(real_ifft(m,axis),fam,p)
                params[(l,fam)].copy_(torch.from_numpy(initial[(l,fam)]+upd));kept+=k;retained+=energy(upd)
            score=text_score(model,inputs,teachers)
            row={'variant':label,'fraction':fraction,'real_coordinates_kept':kept,'energy_retained':retained/sum(energy(d) for d in delta.values()),
              'kl_recovery_fraction':1-score['kl_to_target']/reverted['kl_to_target'],
              'nll_recovery_fraction':(reverted['nll']-score['nll'])/(reverted['nll']-full['nll']),**score}
            rows.append(row);dump('sparse_interventions.json',rows)
            print(label,fraction,score['kl_to_target'],row['kl_recovery_fraction'],flush=True)

if __name__=='__main__':main()
