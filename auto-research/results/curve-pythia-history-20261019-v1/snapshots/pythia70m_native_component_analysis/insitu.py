"""Native-coordinate component atlas. No optimization or neuron permutations."""
import argparse, hashlib, json, pathlib, time, zipfile
import numpy as np
from scipy import fft
from safetensors import safe_open

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
COEFF=ROOT/'coefficients'; COEFF.mkdir(exist_ok=True)
STEPS=json.loads((ROOT/'protocol.json').read_text())['steps']
NAMES=['input_layernorm.weight','input_layernorm.bias','post_attention_layernorm.weight','post_attention_layernorm.bias',
 'attention.query_key_value.weight','attention.query_key_value.bias','attention.dense.weight','attention.dense.bias',
 'mlp.dense_h_to_4h.weight','mlp.dense_h_to_4h.bias','mlp.dense_4h_to_h.weight','mlp.dense_4h_to_h.bias']

def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def en(x): return float(np.sum(np.abs(np.asarray(x))**2,dtype=np.float64))
def tensor(checkpoints,step,layer,name):
    with safe_open(str(checkpoints/f'step{step}'/'model.safetensors'),framework='np') as f:
        return f.get_tensor(f'gpt_neox.layers.{layer}.{name}')

def haar(x):
    a=np.asarray(x,dtype=np.float64).ravel(); n=a.size; details=[]; level=0
    while a.size%2==0:
        details.append((a[0::2]-a[1::2])/np.sqrt(2))
        a=(a[0::2]+a[1::2])/np.sqrt(2);level+=1
    blocks=[a]+details[::-1]; layout=[]; off=0
    layout.append({'kind':'scaling','level':level,'offset':off,'count':a.size,'support':2**level});off+=a.size
    for lev,d in zip(range(level,0,-1),details[::-1]):
        layout.append({'kind':'detail','level':lev,'offset':off,'count':d.size,'support':2**lev});off+=d.size
    assert off==n
    return np.concatenate(blocks),layout

def ihaar(c,layout):
    c=np.asarray(c,dtype=np.float64)
    a=c[:layout[0]['count']].copy()
    for g in layout[1:]:
        d=c[g['offset']:g['offset']+g['count']]
        b=np.empty(a.size*2,np.float64)
        b[0::2]=(a+d)/np.sqrt(2); b[1::2]=(a-d)/np.sqrt(2);a=b
    return a

def packet(x,depth):
    nodes={'':np.asarray(x,dtype=np.float64)}
    for _ in range(depth):
        new={}
        for path,a in nodes.items():
            new[path+'a']=(a[::2]+a[1::2])/np.sqrt(2)
            new[path+'d']=(a[::2]-a[1::2])/np.sqrt(2)
        nodes=new
    return nodes

def ipacket(nodes):
    nodes={k:np.asarray(v,dtype=np.float64).copy() for k,v in nodes.items()}
    while len(next(iter(nodes)))>0:
        new={}
        for parent in sorted(set(k[:-1] for k in nodes)):
            a=nodes[parent+'a'];d=nodes[parent+'d'];b=np.empty(a.size*2)
            b[::2]=(a+d)/np.sqrt(2);b[1::2]=(a-d)/np.sqrt(2);new[parent]=b
        nodes=new
    return nodes['']

def top_indices(v,k=4):
    k=min(k,len(v));ix=np.argpartition(np.abs(v),-k)[-k:]
    return ix[np.argsort(np.abs(v[ix]))[::-1]]

def native_address(name,shape,start,end):
    out={'flat_start':int(start),'flat_end_exclusive':int(end),
         'first_index':[int(x) for x in np.unravel_index(start,shape)],
         'last_index':[int(x) for x in np.unravel_index(end-1,shape)]}
    if name.startswith('attention.query_key_value'):
        width=shape[1] if len(shape)==2 else 1
        rows=[start//width,(end-1)//width]
        out['qkv_first']={'head':int(rows[0]//192),'projection':['q','k','v'][(rows[0]%192)//64],'within_projection':int(rows[0]%64)}
        out['qkv_last']={'head':int(rows[1]//192),'projection':['q','k','v'][(rows[1]%192)//64],'within_projection':int(rows[1]%64)}
    return out

def phase_split(z0,z1,n):
    a=z0.astype(np.complex128);b=z1.astype(np.complex128)
    m=np.full(a.size,2.);m[0]=1.
    if n%2==0:m[-1]=1.
    A=np.abs(a);B=np.abs(b)
    amp=(B-A)**2
    phase=2*np.maximum(0,A*B-(b*np.conj(a)).real)
    total=np.abs(b-a)**2
    return float(np.dot(m,amp)),float(np.dot(m,phase)),float(np.dot(m,total)),float(abs(np.dot(m,amp+phase-total)))

def analyze(checkpoints):
    begin=time.time();summaries=[];scales=[];catalog=[];tracks=[];mapping=[];numerics=[];files=[]
    # Each layer archive retains every coefficient at every checkpoint.
    for layer in range(6):
        start_time=time.time();arrays={'steps':np.array(STEPS,dtype=np.int64)}; layer_meta=[];offset=0
        for tid,name in enumerate(NAMES):
            first=tensor(checkpoints,STEPS[0],layer,name);shape=first.shape;n=first.size
            f=np.empty((len(STEPS),n//2+1),np.complex64)
            h=np.empty((len(STEPS),n),np.float32)
            layout=None
            for si,step in enumerate(STEPS):
                x=tensor(checkpoints,step,layer,name).ravel().astype(np.float64)
                z=fft.rfft(x,norm='ortho',workers=1);c,layout=haar(x)
                f[si]=z;h[si]=c
                if step in [1000,143000]:
                    fr=fft.irfft(f[si].astype(np.complex128),n=n,norm='ortho')
                    hr=ihaar(h[si],layout)
                    denom=en(x)
                    numerics.append({'layer':layer,'tensor':name,'step':step,
                      'fourier_relative_rms':float(np.sqrt(en(fr-x)/(denom+1e-300))),
                      'haar_relative_rms':float(np.sqrt(en(hr-x)/(denom+1e-300))),
                      'fourier_max_abs':float(np.max(np.abs(fr-x))),'haar_max_abs':float(np.max(np.abs(hr-x)))})
            meta={'layer':layer,'tensor':name,'tensor_id':tid,'shape':list(shape),'n':n,
              'layer_curve_start':offset,'layer_curve_end_exclusive':offset+n,'flatten_order':'C',
              'fourier_array':f't{tid:02d}_fourier','haar_array':f't{tid:02d}_haar','haar_layout':layout}
            offset+=n;layer_meta.append(meta);mapping.append(meta)
            arrays[meta['fourier_array']]=f;arrays[meta['haar_array']]=h
            for si in range(1,len(STEPS)):
                a,b=STEPS[si-1:si+1];df=f[si].astype(np.complex128)-f[si-1];dh=h[si].astype(np.float64)-h[si-1]
                amp,phase,total,err=phase_split(f[si-1],f[si],n)
                summaries.append({'layer':layer,'tensor':name,'start':a,'end':b,'interval_steps':b-a,
                   'n':n,'complex_change_energy':total,'amplitude_change_energy':amp,'phase_change_energy':phase,
                   'phase_fraction':phase/total if total else 0.,'phase_split_abs_error':err,
                   'net_rms_change':float(np.sqrt(total/n)),'net_rms_per_training_step':float(np.sqrt(total/n)/(b-a)),
                   'haar_change_energy':en(dh)})
                mode=top_indices(df,4)
                for k in mode:
                    angle=float(np.angle(f[si,k]*np.conj(f[si-1,k]))) if abs(f[si,k])*abs(f[si-1,k])>0 else None
                    catalog.append({'layer':layer,'tensor':name,'start':a,'end':b,'basis':'fourier','index':int(k),
                       'frequency_cycles_per_tensor':int(k),'n':n,'delta_real':float(df[k].real),'delta_imag':float(df[k].imag),
                       'amplitude_before':float(abs(f[si-1,k])),'amplitude_after':float(abs(f[si,k])),
                       'wrapped_phase_change':angle,'change_abs':float(abs(df[k])),
                       'support':'global across this tensor; no unique native weight interval'})
                for g in layout:
                    delta=dh[g['offset']:g['offset']+g['count']]
                    rms=float(np.sqrt(en(delta)/len(delta)))
                    scales.append({'layer':layer,'tensor':name,'start':a,'end':b,**g,'change_energy':en(delta),
                      'rms_change_per_coefficient':rms,'rms_relative_to_tensor':rms/(np.sqrt(total/n)+1e-300)})
                    for k in top_indices(delta,3):
                        ix=g['offset']+int(k);start=int(k)*g['support'];end=start+g['support']
                        catalog.append({'layer':layer,'tensor':name,'start':a,'end':b,'basis':'haar',
                          'kind':g['kind'],'level':g['level'],'index':int(k),'packed_index':ix,
                          'coefficient_before':float(h[si-1,ix]),'coefficient_after':float(h[si,ix]),
                          'signed_change':float(delta[k]),'change_abs':float(abs(delta[k])),
                          **native_address(name,shape,start,end)})
            # Component identities selected once; same identities followed through all 13 states.
            if name in ['attention.query_key_value.weight','mlp.dense_h_to_4h.weight']:
                for event in [(512,1000),(64000,143000)]:
                    i0=STEPS.index(event[0]);i1=STEPS.index(event[1]);df=f[i1]-f[i0]
                    k=int(np.argmax(np.abs(df)))
                    tracks.append({'layer':layer,'tensor':name,'basis':'fourier','selected_interval':list(event),'index':k,
                      'real':f[:,k].real.astype(float).tolist(),'imag':f[:,k].imag.astype(float).tolist(),
                      'amplitude':np.abs(f[:,k]).astype(float).tolist(),'phase':np.angle(f[:,k]).astype(float).tolist()})
                    for lev in [1,6,9,12,15]:
                        g=next((g for g in layout if g['kind']=='detail' and g['level']==lev),None)
                        if g is None: continue
                        d=h[i1,g['offset']:g['offset']+g['count']]-h[i0,g['offset']:g['offset']+g['count']]
                        k=int(np.argmax(np.abs(d)));ix=g['offset']+k
                        tracks.append({'layer':layer,'tensor':name,'basis':'haar','selected_interval':list(event),
                          'kind':'detail','level':lev,'index':k,'packed_index':ix,'values':h[:,ix].astype(float).tolist(),
                          **native_address(name,shape,k*2**lev,(k+1)*2**lev)})
        arrays['metadata_json']=np.array(json.dumps(layer_meta))
        path=COEFF/f'pythia70m_native_layer{layer}_coefficients.npz'
        np.savez_compressed(path,**arrays)
        files.append({'layer':layer,'file':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        dump('tensor_mapping.json',mapping);dump('fourier_component_changes.json',summaries)
        dump('wavelet_scales.json',scales);dump('component_target_catalog.json',catalog)
        dump('component_trajectories.json',tracks);dump('numerical_checks.json',numerics);dump('coefficient_files.json',files)
        print(f'layer {layer}: {offset:,} parameters; archive {path.stat().st_size/1e6:.1f} MB; {time.time()-start_time:.1f}s',flush=True)
        del arrays,f,h
    dump('analysis_environment.json',{'seconds':time.time()-begin,'steps':STEPS,'layers':6,
      'parameter_count':sum(r['n'] for r in mapping),'transforms':'float64 calculation; complex64 Fourier and float32 Haar archives',
      'full_resolution':True,'truncated_coefficients':0,'new_training_steps':0,'neuron_permutations':0,
      'coefficient_archive_bytes':sum(r['bytes'] for r in files)})
    print('FULL-RESOLUTION ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=pathlib.Path,default=ROOT.parent/'pythia70m_spectral_2026-10-17'/'checkpoints')
    analyze(ap.parse_args().checkpoints)
