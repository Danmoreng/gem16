"""Paired, stratified bootstrap for the frozen raw-O boundary/producer chain."""
from pathlib import Path
import argparse,json,random,statistics
p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args();data=json.loads(a.input.read_text());cases=data['cases'];assert len(cases)==12 and all(c['bit_equal'] for c in cases)
rng=random.Random(121115);result={'diagnostic_only':True,'bootstrap':'10000 paired resamples within each fixed layer/phase,seed121115','weighting':'40/3 for each selected local layer,8 global, equal phases','conditions':{}}
for key in ['samples','chain_samples']:
 out={}
 for label,pi,ci in [('warm',0,1),('evicted',2,3)]:
  means=[]
  for c in cases:means.append((c['weight'],statistics.mean(r[pi] for r in c[key]),statistics.mean(r[ci] for r in c[key])))
  parent=sum(w*x for w,x,y in means)/sum(w for w,x,y in means);candidate=sum(w*y for w,x,y in means)/sum(w for w,x,y in means)
  boots=[]
  for _ in range(10000):
   x=y=0.
   for c in cases:
    draws=rng.choices(c[key],k=len(c[key]));x+=c['weight']*statistics.mean(s[pi] for s in draws);y+=c['weight']*statistics.mean(s[ci] for s in draws)
   boots.append(x/y-1)
  boots.sort();out[label]={'parent_weighted_mean_us':parent,'candidate_weighted_mean_us':candidate,'throughput_gain':parent/candidate-1,'lower95':boots[250],'upper95':boots[9749]}
 result['conditions'][key]=out
result['passes_isolated_warm_evicted_gate']=all(x['throughput_gain']>=.01 and x['lower95']>0 for x in result['conditions']['samples'].values())
result['chain_point_estimates_nonnegative']=all(x['throughput_gain']>=0 for x in result['conditions']['chain_samples'].values())
with a.output.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
