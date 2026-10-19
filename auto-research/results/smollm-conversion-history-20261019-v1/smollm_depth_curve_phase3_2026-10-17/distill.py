"""Teacher caches, least-squares initialization, and controlled endpoint training."""
from __future__ import annotations
import copy, json, math, time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from depth_curve import DepthCurveMLP
from evaluation import call_block

ROOT=Path(__file__).resolve().parent


def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


@torch.no_grad()
def teacher_pairs(model,source,windows,label):
    path=ROOT/'cache'/f'{label}_teacher_pairs.pt'
    if path.exists():return torch.load(path,weights_only=True)
    anchors=[];deltas=[];pending=[]
    h1=source.post_attention_layernorm.register_forward_pre_hook(lambda m,a:pending.append(a[0].detach()))
    h2=source.mlp.register_forward_hook(lambda m,a,o:deltas.append(o.detach().reshape(-1,576).clone()))
    for i in range(0,len(windows),4):
        call_block(model,source,windows[i:i+4,:-1])
        anchors.append(pending.pop().reshape(-1,576).clone())
    h1.remove();h2.remove()
    result={'anchor':torch.cat(anchors),'target':torch.cat(deltas)}
    torch.save(result,path)
    return result


@torch.no_grad()
def initialize(source,train,width,knots):
    n=len(train['anchor']);scores=torch.zeros(1536,dtype=torch.float64)
    for i in range(0,n,1024):
        x=source.post_attention_layernorm(train['anchor'][i:i+1024])
        features=F.silu(source.mlp.gate_proj(x))*source.mlp.up_proj(x)
        scores+=features.double().square().sum(0)
    scores*=source.mlp.down_proj.weight.double().square().sum(0)/n
    indices=torch.sort(torch.argsort(scores,descending=True)[:width]).values
    gate=source.mlp.gate_proj.weight[indices].clone();up=source.mlp.up_proj.weight[indices].clone()
    hh=torch.zeros(width,width,dtype=torch.float64);hy=torch.zeros(width,576,dtype=torch.float64)
    for i in range(0,n,1024):
        x=source.post_attention_layernorm(train['anchor'][i:i+1024])
        h=(F.silu(F.linear(x,gate))*F.linear(x,up)).double()
        hh+=h.T@h;hy+=h.T@train['target'][i:i+1024].double()
    hh/=n;hy/=n;ridge=float(hh.diag().mean())*1e-4
    down=torch.linalg.solve(hh+ridge*torch.eye(width,dtype=torch.float64),hy).T.float()
    student=DepthCurveMLP(576,width,knots,source.post_attention_layernorm.weight,source.post_attention_layernorm.variance_epsilon)
    flat=torch.cat([gate.flatten(),up.flatten(),down.flatten()])
    student.coefficients.copy_(flat.repeat(knots,1))
    return student,{'width':width,'knots':knots,'selected_neurons':indices.tolist(),
           'down_refit_ridge':ridge,'down_refit_ridge_relative':1e-4,
           'initialization':'Importance-selected teacher gate/up; down projection refitted by training-only ridge regression. All depth controls initially identical.'}


@torch.no_grad()
def local_metrics(student,data,steps):
    error=0.;target_ss=0.
    for i in range(0,len(data['anchor']),1024):
        target=data['target'][i:i+1024];output=student(data['anchor'][i:i+1024],steps=steps)
        error+=float((output-target).square().sum());target_ss+=float(target.square().sum())
    return {'relative_mse':error/target_ss,'relative_rms_error':math.sqrt(error/target_ss)}


def train_one(student,train,validation,spec,updates=800,batch_size=512,lr=3e-4,pilot=False):
    student=copy.deepcopy(student).train();student._sampled_weights=None
    optimizer=torch.optim.AdamW(student.parameters(),lr=lr,weight_decay=.01,foreach=True)
    generator=torch.Generator().manual_seed(30317 if not pilot else 90917)
    depths=[1,2,4,8] if spec['regime']=='mixed' else [spec.get('steps',1)]
    scale=float(train['target'].square().mean())
    started=time.perf_counter();history=[];best=math.inf;best_state=None;best_update=0
    def validate(update,loss):
        nonlocal best,best_state,best_update
        metrics={str(k):local_metrics(student,validation,k) for k in depths}
        score=sum(v['relative_mse'] for v in metrics.values())/len(metrics)
        row={'update':update,'training_relative_mse':loss,'validation':metrics,'selection_score':score,'elapsed_seconds':time.perf_counter()-started}
        history.append(row)
        if score<best:
            best=score;best_update=update;best_state={k:v.detach().clone() for k,v in student.state_dict().items()}
            if not pilot:save_file(best_state,str(ROOT/'checkpoints'/f"{spec['id']}.safetensors"))
        if not pilot:
            dump(ROOT/'results'/f"training_{spec['id']}.json",{'spec':spec,'configuration':student.configuration(),
                    'history':history,'best_update':best_update,'selection':'Lowest validation endpoint MSE; never test metrics.'})
            print('TRAIN',spec['id'],update,'val relative RMS',round(math.sqrt(score),5),'seconds',round(time.perf_counter()-started,1),flush=True)
    validate(0,None)
    for update in range(1,updates+1):
        ids=torch.randint(len(train['anchor']),(batch_size,),generator=generator)
        steps=depths[(update-1)%len(depths)]
        warm=min(1.,update/min(40,max(1,updates//5)))
        progress=max(0.,(update-40)/max(1,updates-40))
        rate=lr*warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        optimizer.param_groups[0]['lr']=rate;optimizer.zero_grad(set_to_none=True)
        prediction=student(train['anchor'][ids],steps=steps)
        loss=(prediction-train['target'][ids]).square().mean()/scale
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss: {spec}, update {update}')
        loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.0);optimizer.step()
        if update%100==0 or update==updates:validate(update,float(loss.detach()))
    student.load_state_dict(best_state);student.eval()
    return student,{'spec':spec,'configuration':student.configuration(),'updates':updates,'batch_size':batch_size,
       'token_presentations':updates*batch_size,'peak_learning_rate':lr,'weight_decay':.01,'gradient_clip_norm':1.,
       'field_token_evaluations':batch_size*sum(depths[(i-1)%len(depths)] for i in range(1,updates+1)),
       'linear_forward_training_macs_excluding_weight_assembly':batch_size*sum(depths[(i-1)%len(depths)] for i in range(1,updates+1))*3*student.hidden_size*student.width,
       'best_update':best_update,'best_validation_relative_mse':best,'history':history,'elapsed_seconds':time.perf_counter()-started}
