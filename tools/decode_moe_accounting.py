"""Offline source-defined MoE fork/join accounting; not a speedup model."""
from collections import Counter

def branch_of(row):
 name=row['name'];scope=row['capture_scope']
 if 'target_shared_moe' in scope or row['category'] in ['shared_mlp_gateup','shared_mlp_down'] or name=='QuantizeActivationReferenceKernel':return 'shared'
 return 'routed'

def reconstruct(rows):
 norm=rows[0];post=rows[-1]
 shared=[r for r in rows[1:-1] if branch_of(r)=='shared'];routed=[r for r in rows[1:-1] if branch_of(r)=='routed']
 assert shared and routed
 shared_end=max(r['end'] for r in shared);routed_end=max(r['end'] for r in routed)
 assert max(shared_end,routed_end)<=post['start'],(shared_end,routed_end,post['start'])
 winner='routed' if routed_end>=shared_end else 'shared';critical=([norm]+(routed if winner=='routed' else shared)+[post])
 ordered=sorted(critical,key=lambda r:r['start'])
 assert all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:])), 'PDL/other within-branch overlap requires a more detailed DAG'
 cost=Counter()
 for r in critical:cost[r['category']]+=r['end']-r['start']
 span=post['end']-norm['start'];gap=span-sum(cost.values());assert gap>=0
 cost['moe_observed_dependency_or_scheduling_gaps']+=gap
 assert sum(cost.values())==span
 return {'winner':winner,'routed_tail_after_shared_ns':routed_end-shared_end,'span_ns':span,'critical_category_ns':dict(cost)}
