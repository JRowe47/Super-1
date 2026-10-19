import argparse,gc,hashlib,json,math,struct,time
from pathlib import Path
import numpy as np
from scipy.fft import rfft,irfft

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
class Arrays:
    def __init__(self,path):
        self.path=Path(path)
        with self.path.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];self.h=json.loads(f.read(n))
        self.offset=8+n
    def get(self,name):
        h=self.h[name];typ={'F32':'<f4','BF16':'<u2'}[h['dtype']]
        lo,hi=h['data_offsets'];assert hi-lo==math.prod(h['shape'])*np.dtype(typ).itemsize
        x=np.memmap(self.path,dtype=typ,mode='r',offset=self.offset+lo,shape=tuple(h['shape']))
        return (x.astype(np.uint32)<<16).view(np.float32) if h['dtype']=='BF16' else np.asarray(x)

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')

class Source:
    def __init__(self,src):
        self.src=src;self.orig=Arrays(src/'reference_model/model.safetensors')
        self.final={a:Arrays(src/'checkpoints'/f'{a}.safetensors') for a in ['retained','interpolated']}
        self.slopes={s:Arrays(src/'reference/phase4_checkpoints'/f'window_{s}.safetensors') for s in [4,12]}
        self.names=sorted(n.split('.',2)[2] for n in self.final['retained'].h if n.startswith('4.0.'))
        self.map=[];i=0
        while i<30:
            if i in [4,12]:self.map.extend([(i,j) for j in range(3)]);i+=4
            else:self.map.append(i);i+=1
        self.p=sum(math.prod(self.orig.h['model.layers.0.'+n]['shape']) for n in self.names)
    def tensor(self,p,name,arm='retained',trained=False):
        if isinstance(p,int):return self.orig.get(f'model.layers.{p}.{name}')
        s,j=p
        if trained:return self.final[arm].get(f'{s}.{j}.{name}')
        if arm=='retained':return self.orig.get(f'model.layers.{s+[0,2,3][j]}.{name}')
        t=(j+.5)*4/3-.5;lo=math.floor(t);u=t-lo
        h=(2*u**3-3*u**2+1,u**3-2*u**2+u,-2*u**3+3*u**2,u**3-u**2)
        m=self.slopes[s].get(name.replace('.','__'))
        return h[0]*self.orig.get(f'model.layers.{s+lo}.{name}')+h[1]*m[lo]+h[2]*self.orig.get(f'model.layers.{s+lo+1}.{name}')+h[3]*m[lo+1]
    def wave(self,kind,column_major=False):
        pos=list(range(30)) if kind=='original30' else self.map
        x=np.empty(len(pos)*self.p,dtype=np.float32);offset=0;manifest=[]
        arm='interpolated' if kind.startswith('interpolated') else 'retained'
        for block,p in enumerate(pos):
            for name in self.names:
                v=self.tensor(p,name,arm,kind.endswith('_final'))
                if kind.endswith('_delta'):v=self.tensor(p,name,arm,True)-v
                flat=v.ravel(order='F' if column_major else 'C');n=len(flat)
                x[offset:offset+n]=flat
                manifest.append({'block':block,'source':p,'family':name,'start':offset,'end':offset+n,'shape':list(v.shape)})
                offset+=n
        return x,manifest

def tree(spec,n,depth=6):
    # Allocate energy only once. Double non-endpoint real-FFT bins.
    power=spec.real.astype(np.float64)**2+spec.imag.astype(np.float64)**2
    power*=2;power[0]*=.5
    if n%2==0:power[-1]*=.5
    leaves=[]
    def walk(lo,hi,level,index):
        energy=float(power[lo:hi].sum());dof=2*(hi-lo)-(lo==0)-(hi==len(power) and n%2==0)
        leaves.append({'level':level,'index':index,'bin_start':lo,'bin_stop':hi,'cycles_per_weight_start':lo/n,'cycles_per_weight_stop':min(.5,hi/n),'energy':energy,'real_dof':int(dof)})
        if level<depth:
            mid=(lo+hi)//2;walk(lo,mid,level+1,index*2);walk(mid,hi,level+1,index*2+1)
    walk(0,len(power),0,0)
    total=leaves[0]['energy']
    for r in leaves:r['fraction']=r['energy']/total
    return {'n':n,'bins':len(spec),'total_energy':total,'nodes':leaves}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase6',type=Path,required=True);args=ap.parse_args()
    source=Source(args.phase6);out={};overall=time.perf_counter()
    variants=['original30','retained_initial','retained_final','retained_delta','interpolated_initial','interpolated_final','interpolated_delta','retained_delta_column_major']
    for label in variants:
        started=time.perf_counter();column=label.endswith('_column_major');kind=label.replace('_column_major','')
        x,manifest=source.wave(kind,column);t=time.perf_counter();f=rfft(x,norm='ortho',workers=4)
        transform_seconds=time.perf_counter()-t;record=tree(f,len(x));record['forward_fft_seconds']=transform_seconds
        record['input_squared_norm']=float(np.dot(x.astype(np.float64),x.astype(np.float64)))
        record['parseval_relative_error']=abs(record['total_energy']/record['input_squared_norm']-1)
        if label=='retained_delta':
            dump(OUT/'waveform_map.json',manifest)
            where=next(r for r in manifest if r['block']==4 and r['family']=='self_attn.q_proj.weight')
            zoom=np.arange(where['start']+2048,where['start']+2560)
            components=[];reconstructed=np.zeros_like(x);edges=np.linspace(0,len(f),9,dtype=np.int64)
            for i in range(8):
                masked=np.zeros_like(f);masked[edges[i]:edges[i+1]]=f[edges[i]:edges[i+1]]
                c=irfft(masked,n=len(x),norm='ortho',workers=4);reconstructed+=c
                components.append(c[zoom].copy());del masked,c
            record['eight_component_max_abs_reconstruction_error']=float(np.max(np.abs(reconstructed-x)))
            record['eight_component_relative_rms_error']=float(np.sqrt(np.sum((reconstructed.astype(np.float64)-x)**2)/record['input_squared_norm']))
            record['excerpt']={'start':int(zoom[0]),'stop':int(zoom[-1]+1),'family':'self_attn.q_proj.weight','block28':4}
            np.savez_compressed(OUT/'component_excerpt.npz',indices=zoom,original=x[zoom],components=np.stack(components),summed=reconstructed[zoom])
            del reconstructed
        record['seconds']=time.perf_counter()-started;out[label]=record
        dump(OUT/'global_spectra.json',out)
        print(json.dumps({'variant':label,'seconds':round(record['seconds'],2),'FFT_seconds':round(transform_seconds,2),'parseval_relative_error':record['parseval_relative_error']}),flush=True)
        del x,f;gc.collect()
    dump(OUT/'global_environment.json',{'total_seconds':time.perf_counter()-overall,'precision':'float32 waveforms, complex64 FFT, float64 energy accumulation','axis':'flattened block-major waveform, not model depth','phase6_source':str(args.phase6),'weights_per_block':source.p,'ordering_control':'Retained update repeated with column-major flattening; decoding restores exactly the same tensors.'})

if __name__=='__main__':main()
