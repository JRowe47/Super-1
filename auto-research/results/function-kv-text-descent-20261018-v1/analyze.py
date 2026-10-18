"""Verify frozen experiment evidence and record separate quality/memory gates."""
import hashlib,json,math,statistics,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2];C=json.loads((H/'CONFIG.json').read_text());O=H/'execution'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def contrast(gaps):
    mean=statistics.mean(gaps);sem=statistics.stdev(gaps)/math.sqrt(len(gaps));upper=mean+C['t95_df2']*sem
    return {'paired_ce_gaps':gaps,'mean_ce_gap':mean,'paired_t95':[mean-C['t95_df2']*sem,upper],'geometric_ppl_ratio':math.exp(mean),'passes_quality_gate':upper<=math.log1p(C['quality_ppl_tolerance']) and max(gaps)<=math.log1p(C['quality_ppl_tolerance'])}
def main():
    f=json.loads((O/'FREEZE.json').read_text());r=json.loads((O/'RESULT.json').read_text());t=json.loads((O/'TERMINAL.json').read_text());assert t['status']=='COMPLETED'
    manifest=json.loads((H/'SOURCE_MANIFEST.json').read_text());assert all(sha(ROOT/p)==s for p,s in manifest.items())
    for row in f['history']:assert sha(O/row['checkpoint'])==row['sha256']
    rows=r['rows'];memdir=O/'memory';memdir.mkdir(exist_ok=False)
    for seed in C['seeds']:
        for length in C['memory_lengths']:
            for arm in ['exact',*C['candidates']]:
                dest=memdir/f'{arm}-{seed}-{length}.json'
                subprocess.run([sys.executable,'-B',str(H/'run.py'),'memory','--arm',arm,'--seed',str(seed),'--length',str(length),'--destination',str(dest)],cwd=ROOT,check=True)
                print(json.dumps(json.loads(dest.read_text())),flush=True)
    memory=[json.loads(p.read_text()) for p in memdir.glob('*.json')]
    def metric(seed,arm,variant='selected',key='ce'):
        matches=[v for v in rows if v['seed']==seed and v['arm']==arm and (arm in ('nano','curve') or v['variant']==variant)]
        if variant=='zero' and not matches:matches=[v for v in rows if v['seed']==seed and v['arm']==arm and v.get('round')==0]
        assert matches
        return matches[0]['confirmation'][key]
    summary={'means':{},'contrasts':{},'selected_capacity':f['selected_capacity'],'selection_feasible':f['selection_feasible'],'memory':memory,'verified_round_checkpoints':len(f['history']),'verified_source_hashes':len(manifest),'training':t,'limitations':['Single-book byte pilot, not SOTA or nanoGPT speedrun replication.','128-byte quality only; larger context measurements are mechanics, not recall or language quality.','Identical base target exposure, not equal parameter count or training FLOPs; hierarchy receives extra supervised and distillation training.','Peak RSS includes interpreter/allocator/runtime. Cache-only and retained parameter+cache bytes are separately reported.']}
    for arm in ('nano','curve',*C['candidates']):
        summary['means'][arm]={'ce':statistics.mean(metric(s,arm) for s in C['seeds']),'tail_ce':statistics.mean(metric(s,arm,key='tail_ce') for s in C['seeds'])}
        if arm in C['candidates']:summary['means'][arm]['zero_pool_ce']=statistics.mean(metric(s,arm,'zero') for s in C['seeds'])
    for arm in ('curve',*C['candidates']):
        for ref in ('nano','curve'):
            if arm!=ref:summary['contrasts'][arm+'_minus_'+ref]=contrast([metric(s,arm)-metric(s,ref) for s in C['seeds']])
        if arm in C['candidates']:summary['contrasts'][arm+'_trained_minus_zero']=contrast([metric(s,arm)-metric(s,arm,'zero') for s in C['seeds']])
    choice=f['selected_capacity'];quality=all(summary['contrasts'][choice+'_minus_'+ref]['passes_quality_gate'] for ref in ('nano','curve'))
    ratios=[]
    for s in C['seeds']:
        for length in C['memory_lengths']:
            exact=next(v for v in memory if v['seed']==s and v['length']==length and v['arm']=='exact');h=next(v for v in memory if v['seed']==s and v['length']==length and v['arm']==choice)
            ratios.append({'seed':s,'length':length,'cache_reduction':exact['logical_cache_bytes']/h['logical_cache_bytes'],'retained_weights_plus_cache_reduction':(exact['logical_cache_bytes']+exact['parameter_bytes'])/(h['logical_cache_bytes']+h['parameter_bytes']),'peak_rss_reduction':exact['peak_rss_bytes']/h['peak_rss_bytes']})
    memory_pass=all(v['peak_rss_reduction']>=C['total_memory_reduction_target'] for v in ratios)
    summary.update(quality_gate_pass=quality,total_memory_gate_pass=memory_pass,promoted=quality and memory_pass,selected_memory_ratios=ratios)
    save(O/'SUMMARY.json',summary)
    entries=[]
    for row in f['hierarchy']:
        entries.append({'id':C['run_id']+'/'+row['arm']+'/'+str(row['seed']),'use_case':'text/alice-utf8-byte','active':False,'status':'EXPERIMENTAL_EXPLICIT_OPT_IN','capacity':C['candidates'][row['arm']],'context':C['context'],'execution_depth':C['layers'],'base_checkpoint':{'path':str((O/row['base_checkpoint']).relative_to(ROOT)),'sha256':row['base_sha256']},'pool_checkpoint':{'path':str((O/row['checkpoint']).relative_to(ROOT)),'sha256':row['sha256']},'source_manifest':{'path':str((H/'SOURCE_MANIFEST.json').relative_to(ROOT)),'sha256':sha(H/'SOURCE_MANIFEST.json')},'config':{'path':str((H/'CONFIG.json').relative_to(ROOT)),'sha256':sha(H/'CONFIG.json')},'selection':row['selection'],'confirmation_ce':metric(row['seed'],row['arm']),'limitations':summary['limitations']})
    save(H/'LIBRARY.json',{'entries':entries,'automatic_promotion':False,'reason':'Retain explicit measured profiles; no broad-quality or total-memory success inferred from cache reduction.'})
    save(O/'VERIFICATION.json',{'status':'PASS','source_hashes':len(manifest),'checkpoint_hashes':len(f['history']),'confirmation_duplicate_equality':'Asserted for every row during train.py execution','isolated_memory_workers':len(memory),'summary_sha256':sha(O/'SUMMARY.json'),'library_sha256':sha(H/'LIBRARY.json'),'quality_gate_pass':quality,'total_memory_gate_pass':memory_pass})
    print(json.dumps({k:v for k,v in summary.items() if k!='memory'},indent=2))
if __name__=='__main__':main()
