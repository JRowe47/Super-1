"""Print the complete behavioral effect of a saved native component."""
import argparse,json
import numpy as np
from calibrate import MODELS
ap=argparse.ArgumentParser()
ap.add_argument('--task',choices=['readout','parity','context','joint'],default='joint')
ap.add_argument('--seed',type=int,default=1);ap.add_argument('--component',type=int,default=197)
ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=800)
ap.add_argument('--direction',choices=['rollback','transplant'],default='rollback')
a=ap.parse_args();z=np.load(MODELS/f'{a.task}_seed{a.seed}.npz')
catalog=json.loads(str(z['catalog_json']));c=catalog[a.component]
intervals=z['intervals'].tolist();pi=intervals.index([a.start,a.end])
si=z['steps'].tolist().index(a.end if a.direction=='rollback' else a.start)
before=z['original_logits'][si];after=z[f'{a.direction}_logits'][pi,a.component];target=z['targets']
print(json.dumps(dict(component=c,interval=[a.start,a.end],direction=a.direction,
    update_norm=float(z['component_delta_norm'][pi,a.component]),inputs=z['inputs'].tolist(),targets=target.tolist(),
    before_logits=before.tolist(),after_logits=after.tolist(),
    before_accuracy=np.mean((before>0)==(target>0),axis=0).tolist(),after_accuracy=np.mean((after>0)==(target>0),axis=0).tolist(),
    behavior_feature_changes=((np.tanh(after/2)-np.tanh(before/2)).T@z['walsh_basis']/16).tolist()),indent=2))
