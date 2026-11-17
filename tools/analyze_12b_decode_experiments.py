"""Stratified paired bootstrap of explicit12B operator screens (not E2E)."""
from pathlib import Path
import argparse,json,random,statistics
p=argparse.ArgumentParser(description=__doc__);p.add_argument('screen',type=Path);p.add_argument('output',type=Path);p.add_argument('--fp8-layer-weights',action='store_true');p.add_argument('--draws',type=int,default=10000);a=p.parse_args()
d=json.loads(a.screen.read_text());cases=d['cases'];assert cases and all(c['bit_equal'] for c in cases)
rng=random.Random(121115);result={'diagnostic_only':True,'bootstrap':'paired sample resampling within each fixed phase/operator;10000 draws;seed121115','families':{}}
for family in (['all','QKV','O'] if a.fp8_layer_weights else ['all']):
 selected=[c for c in cases if family=='all' or c['family']==family]
 weights=[(8.0 if c['layer']==47 else 40.0/3) if a.fp8_layer_weights else 1.0 for c in selected]
 out={}
 for pi,ci,label in [(0,1,'warm'),(2,3,'evicted')]:
  parent=sum(w*statistics.mean(s[pi] for s in c['samples']) for w,c in zip(weights,selected));candidate=sum(w*statistics.mean(s[ci] for s in c['samples']) for w,c in zip(weights,selected))
  boots=[]
  for _ in range(a.draws):
   pv=cv=0.0
   for w,c in zip(weights,selected):
    samples=rng.choices(c['samples'],k=len(c['samples']));pv+=w*statistics.mean(s[pi] for s in samples);cv+=w*statistics.mean(s[ci] for s in samples)
   boots.append(pv/cv-1)
  boots.sort();out[label]={'parent_weighted_mean_us':parent/sum(weights),'candidate_weighted_mean_us':candidate/sum(weights),'throughput_gain':parent/candidate-1,'lower_95':boots[int(a.draws*.025)],'upper_95':boots[int(a.draws*.975)-1]}
 out['passes_both_cache_conditions']=all(x['throughput_gain']>=.01 and x['lower_95']>0 for x in (out['warm'],out['evicted']))
 result['families'][family]=out
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
