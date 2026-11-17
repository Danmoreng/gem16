"""AP1 offline native 26B decode accounting. Historical records are never rewritten.

Nsight SQLite is read-only; outputs are exclusive-created. Category wall times
are interval unions, not sums of overlapping kernels or causal speedup budgets.
Unknown template dimensions remain explicit rather than defaulting to local.
The 256-local/512-global mapping belongs to this 26B profile, not to 12B.
"""
from collections import Counter,defaultdict
import bisect,csv,json,re,sqlite3,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
MASK32=(1<<32)-1

def attention_kind(name, demangled=""):
 # Short names omit the template head dimension of the generic merge.
 generic=next((kernel for kernel in ("MergeOnlineDecodeAttentionKernel", "SplitOnlineDecodeAttentionFp8Kernel") if kernel in name),None)
 if generic:
  match=re.search(re.escape(generic)+r"<\s*(?:\(int\))?(256|512)\s*,",demangled)
  return {"256":"local","512":"global"}.get(match.group(1)) if match else None
 if "Global" in name:return "global"
 if "Local" in name:return "local"
 return None

def classify(name,scope,gridx,phase,demangled=""):
 if name.startswith('Ap1'):return 'diagnostic_observer'

 if name=='ArgmaxKernel':return 'selection_extra_masked_argmax'
 if name.startswith('Sm120ProjectionArgmaxCandidates'):return 'assistant_output_head'
 if phase.startswith('assistant'):
  if 'Argmax' in name:return 'assistant_output_selection'
  if 'Attention' in name:return 'assistant_attention'
  if 'Projection' in name:return 'assistant_projections'
  if 'Norm' in name:return 'assistant_norms'
  return 'assistant_input_transform_control'
 if 'MmaW4A8Projection' in name:return 'trellis_gateup_codec_mma_inverse' if gridx==11 else 'trellis_down_codec_mma_inverse'
 if 'SelectedInputTransform' in name:return 'trellis_gateup_input_transform'
 if 'GatedGeluDownTransform' in name:return 'trellis_gelu_down_transform_quantization'
 if 'SlotOrderedReduction' in name:return 'moe_expert_reduction'
 if 'Router' in name and ('Projection' in name or 'TopK' in name):return 'moe_router_projection_topk'
 if 'MoeInputNorms' in name:return 'moe_input_norms_router_transform'
 if 'MoeFusedPostNormResidual' in name:return 'moe_postnorm_residual'
 if 'BackupAppendKv' in name or 'CompactRestore' in name or 'CommitMtpKv' in name:return 'kv_backup_restore_commit'
 if 'DecodeKvEpilogue' in name or 'MtpKvEpilogue' in name:return 'kv_epilogue'
 if 'Attention' in name:
  kind=attention_kind(name,demangled)
  return 'attention_'+kind+'_and_merge' if kind else 'attention_unknown'
 if 'SoftcapArgmax' in name or name=='FinalArgmaxKernel' or 'Logit' in name:return 'target_softcap_selection_finite'
 if 'Sm120FusedGateUp' in name:return 'shared_mlp_gateup'
 if 'Sm120DirectProjection' in name or 'GroupedQkvProjection' in name:
  if 'target_head' in scope or gridx==8192:return 'target_output_head'
  if 'target_shared_moe' in scope or phase.endswith('_moe'):return 'shared_mlp_down'
  return 'attention_qkv_o_projections'
 if 'QuantizeToken' in name or 'QuantizeActivation' in name:
  if 'target_routed_moe' in scope or phase.endswith('_moe'):return 'moe_activation_quantization'
  if 'target_head' in scope or phase.endswith('_head'):return 'target_head_norm_quantization'
  return 'attention_norm_quantization'
 if 'RmsNorm' in name:
  return 'target_head_norm_quantization' if phase.endswith('_head') or 'target_head' in scope else 'attention_norm_residual'
 if 'Rotary' in name:return 'attention_rotary'
 if 'Embedding' in name:return 'embedding'
 if any(s in name for s in ['Mtp','DecodeControl','Prediction','ControlledAssistant']):return 'mtp_decode_control_finite_commit'
 if name=='RoundBf16Kernel':return 'attention_rounding'
 return 'unclassified:'+name

def sweep(ops,apis,start,end):
 events=defaultdict(list)
 for a,b,category in ops:
  a=max(a,start);b=min(b,end)
  if a<b:events[a].append(('gpu',category,1));events[b].append(('gpu',category,-1))
 for a,b,name in apis:
  a=max(a,start);b=min(b,end)
  if a<b:events[a].append(('cpu',name,1));events[b].append(('cpu',name,-1))
 events[start];events[end]
 active={'gpu':Counter(),'cpu':Counter()};exclusive=Counter();inclusive=Counter();overlap=Counter();idle=Counter();busy=0
 prev=start
 for position in sorted(events):
  dt=position-prev
  g=[k for k,v in active['gpu'].items() if v>0]
  cpu=[k for k,v in active['cpu'].items() if v>0]
  if g:
   busy+=dt
   for k in g:inclusive[k]+=dt
   if len(g)==1:exclusive[g[0]]+=dt
   else:overlap[' + '.join(sorted(g))]+=dt
  else:
   if any('Memcpy' in n for n in cpu):label='gpu_idle_during_host_copy_api'
   elif any('Launch' in n for n in cpu):label='gpu_idle_during_host_launch_api'
   elif any('Synchronize' in n for n in cpu):label='gpu_idle_during_host_sync_api'
   elif any('StreamQuery' in n for n in cpu):label='gpu_idle_during_host_poll_api'
   elif cpu:label='gpu_idle_during_other_cuda_api'
   else:label='gpu_idle_between_recorded_apis'
   idle[label]+=dt
  for typ,key,delta in events[position]:active[typ][key]+=delta
  prev=position
 assert busy+sum(idle.values())==end-start
 assert sum(exclusive.values())+sum(overlap.values())==busy
 return {'wall_ms':(end-start)/1e6,'gpu_busy_union_ms':busy/1e6,'gpu_idle_ms':{k:v/1e6 for k,v in idle.items()},'gpu_exclusive_category_ms':{k:v/1e6 for k,v in exclusive.items()},'gpu_inclusive_category_ms':{k:v/1e6 for k,v in inclusive.items()},'gpu_overlap_category_pairs_ms':{k:v/1e6 for k,v in overlap.items()},'gpu_overlap_ms':sum(overlap.values())/1e6}

def analyze(path):
 con=sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
 strings=dict(con.execute('select id,value from StringIds'))
 kernels=[dict(r) for r in con.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
 for k in kernels:k['name']=strings[k['shortName']];k['demangled']=strings[k['demangledName']]
 api=[{**dict(r),'name':strings[r['nameId']]} for r in con.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME order by start')]
 ranges=[{**dict(r),'name':r['text'] or strings.get(r['textId'],'')} for r in con.execute('select * from NVTX_EVENTS')]
 nodeevents=[dict(r) for r in con.execute('select * from CUDA_GRAPH_NODE_EVENTS')]
 nodes={r['graphNodeId']:r for r in nodeevents if r['originalGraphNodeId'] is None}
 # The source capture graph and flattened executable have different IDs for
 # conditional bodies. Calibrate a contiguous offset from the distinct init
 # markers, rather than treating live graphLaunch correlation as a capture span.
 d2=any(k['name']=='AdvanceMtpD2ChainKernel' for k in kernels)
 mapping={};mapping_info={}
 if d2:
  candidates=[]
  for nodeid,n in nodes.items():
   if nodeid&MASK32:continue
   scopes=[r for r in ranges if r['end'] is not None and r['start']<=n['start']<=r['end'] and r['globalTid']==n['globalTid']]
   if any(r['name']=='gem16.mtp.assistant_proposal' for r in scopes):candidates.append(n)
  source=max(candidates,key=lambda n:n['start'])['graphNodeId']>>32
  d2nodes={i&MASK32:n for i,n in nodes.items() if i>>32==source}
  base=min(k['graphNodeId']&MASK32 for k in kernels if k['name']=='InitializeControlledAssistantKernel')
  for index,n in d2nodes.items():mapping[base+index]=n
  tailbases=[k['graphNodeId']&MASK32 for k in kernels if k['name']=='InitializeMtpOrdinaryTailKernel']
  # Tail body is the next source conditional graph. Its extent is checked
  # against the observed ordinary-tail marker when that branch actually runs.
  tail_source=source+1
  tailnodes={i&MASK32:n for i,n in nodes.items() if i>>32==tail_source}
  tailbase=base+len(d2nodes)
  if tailbases:assert set(tailbases)=={tailbase},(tailbases,tailbase)
  for index,n in tailnodes.items():mapping[tailbase+index]=n
  mapping_info={'source_d2_graph_id':source,'source_d2_nodes':len(d2nodes),'flattened_d2_start':base,'source_tail_graph_id':tail_source,'source_tail_nodes':len(tailnodes),'flattened_tail_start':tailbase,'tail_anchor_observed':bool(tailbases),'assumption':'contiguous flattened conditional-body nodes; calibrated by init markers, capture timestamps and tail extent'}
 # Identify the whole request without using an NVTX range whose Pop occurred
 # outside collection (some Nsight exports leave that root in another timebase).
 launches=[r for r in api if r['name'].startswith('cudaGraphLaunch') and r['start']>=0]
 assert launches
 start=min(r['start'] for r in launches)
 if not d2:
  selections=[r for r in ranges if r['name']=='gem16.mtp.selection_sampling' and r['start']>=0 and r['end'] is not None and r['end']<max(k['end'] for k in kernels)+10_000_000]
  assert selections
  end=max(r['end'] for r in selections)
 else:
  relevant=[r for r in api if r['start']>=start and ('Memcpy' in r['name'] or 'StreamQuery' in r['name'])]
  end=max([r['end'] for r in relevant]+[k['end'] for k in kernels])
 apis=[(r['start'],r['end'],r['name']) for r in api if r['start']>=start and r['end']<=end]
 ordinary_state={};layer=0;phase='ordinary_attention'
 # Infer ordinary graph stages from ordered main-stream boundary kernels.
 main_stream=Counter(k['streamId'] for k in kernels).most_common(1)[0][0]
 for k in kernels:
  if k['name']=='TiledEmbeddingLookupControlledKernel':layer=0;phase='ordinary_attention'
  if k['name']=='MoeInputNormsRouterTransformTrellis35Kernel':phase='ordinary_moe'
  if k['graphNodeId'] is not None:ordinary_state[k['graphNodeId']]=(phase,layer)
  if k['name']=='MoeFusedPostNormResidualKernel':
   layer+=1;phase='ordinary_head' if layer==30 else 'ordinary_attention'
 # A secondary-stream Shared MLP can start before primary routing work; use
 # source scope where present, otherwise classify dense shared kernels by name.
 created_scopes={}
 target_attention=[r for r in ranges if r['name']=='gem16.mtp.target_attention' and r['start']<0 and r['end'] is not None]
 if d2:
  source_times=[n['start'] for i,n in nodes.items() if i>>32==mapping_info['source_d2_graph_id']]
  target_attention=sorted([r for r in target_attention if min(source_times)<=r['start']<=max(source_times)],key=lambda r:r['start'])
  assert len(target_attention)==30,len(target_attention)
  for index,n in mapping.items():
   scopes=sorted([r for r in ranges if r['end'] is not None and r['start']<=n['start']<=r['end'] and r['globalTid']==n['globalTid'] and r['name'].startswith('gem16.mtp.')],key=lambda r:r['end']-r['start'])
   labels=[r['name'] for r in scopes]
   l=next((i for i,r in enumerate(target_attention) if r['start']<=n['start']<=r['end']),None)
   created_scopes[index]=(labels,l)
 ops=[];category_sum=Counter();category_count=Counter();kernel_rows=[];unknown=Counter();node_map_hits=0
 for k in kernels:
  if k['end']<=start or k['start']>=end:continue
  scope='';p='ordinary_attention';l=None
  if d2:
   low=k['graphNodeId']&MASK32 if k['graphNodeId'] is not None else -1
   if low in created_scopes:
    labels,l=created_scopes[low];scope=';'.join(labels);node_map_hits+=1
    if 'assistant_proposal' in scope:p='assistant'
    elif 'target_shared_moe' in scope or 'target_routed_moe' in scope or 'target_moe' in scope:p='target_moe'
    elif 'target_head' in scope:p='target_head'
    elif 'target_attention' in scope:p='target_attention'
    elif low>=mapping_info['flattened_tail_start']:p=ordinary_state.get(k['graphNodeId'],('ordinary_attention',None))[0]
    else:p='target_control'
   else:p='target_control'
  elif k['graphNodeId'] in ordinary_state:p,l=ordinary_state[k['graphNodeId']]
  category=classify(k['name'],scope,k['gridX'],p,k['demangled'])
  if category.startswith('unclassified:') or category=='attention_unknown':unknown[k['demangled']]+=min(k['end'],end)-max(k['start'],start)
  ops.append((k['start'],k['end'],category));category_sum[category]+=min(k['end'],end)-max(k['start'],start);category_count[category]+=1
  kernel_rows.append({**k,'category':category,'phase':p,'layer':l,'capture_scope':scope})
 tables={r[0] for r in con.execute("select name from sqlite_master where type='table'")}
 for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
  if table not in tables:continue
  for r in con.execute('select * from '+table):
   if r['end']<=start or r['start']>=end:continue
   category='gpu_memset'
   if table.endswith('MEMCPY'):
    category={1:'host_to_device_copy',2:'device_to_host_copy',8:'device_to_device_copy'}.get(r['copyKind'],'gpu_copy_kind_'+str(r['copyKind']))
   ops.append((r['start'],r['end'],category))
 result=sweep(ops,apis,start,end)
 result.update({'input_sqlite':str(path),'mode':'d2' if d2 else 'ordinary','window_ns':[start,end],'kernel_count':len(kernel_rows),'kernel_sum_ms':sum(category_sum.values())/1e6,'kernel_categories':{k:{'count':category_count[k],'sum_ms':v/1e6} for k,v in category_sum.items()},'unclassified_kernel_sum_ms':{k:v/1e6 for k,v in unknown.items()},'capture_mapping':mapping_info,'mapped_kernel_instances':node_map_hits,'api_call_counts':dict(Counter(n for a,b,n in apis)),'api_sum_ms':dict((n,sum(b-a for a,b,name in apis if name==n)/1e6) for n in sorted({n for a,b,n in apis})),'method':'interval union and exclusive-category/overlap accounting. Not a counterfactual causal critical-path speedup. Captured APIs excluded from request runtime.'})
 if not d2:
  argmax=[k for k in kernel_rows if k['name']=='ArgmaxKernel']
  result['extra_argmax_us']=[(k['end']-k['start'])/1000 for k in argmax]
  result['ordinary_tokens']=sum(k['name']=='TiledEmbeddingLookupControlledKernel' for k in kernel_rows)
 else:
  result['verification_groups']=sum(k['name']=='AdvanceMtpD2ChainKernel' for k in kernel_rows)
  result['ordinary_tail_tokens']=sum(k['name']=='FinalizeMtpOrdinaryTailKernel' for k in kernel_rows)
  result['group_windows_ns']=[k['start'] for k in kernel_rows if k['name']=='InitializeControlledAssistantKernel']
 # Full traces are post-processed at beginning/middle/end; all inference executes.
 boundaries=result.get('group_windows_ns') or [r['start'] for r in launches]
 boundaries=sorted(boundaries)
 windows={}
 for label,index in [('begin',0),('middle',len(boundaries)//2),('end',max(0,len(boundaries)-8))]:
  a=boundaries[index];b=boundaries[index+8] if index+8<len(boundaries) else end
  windows[label]={'first_unit_index':index,'unit_count':min(8,len(boundaries)-index),**sweep(ops,apis,a,b)}
 result['representative_windows']=windows
 con.close()
 return result,kernel_rows

if __name__=='__main__':
 for arg in sys.argv[1:]:
  path=Path(arg);result,kernels=analyze(path)
  target=path.with_suffix('.cost.json');assert not target.exists()
  target.write_text(json.dumps(result,indent=2)+'\n')
  # Retain per-kernel source mapping for validating layer/phase classification.
  fields=['start','end','demangled','name','category','phase','layer','capture_scope','streamId','graphNodeId','gridX','gridY','blockX','registersPerThread']
  with path.with_suffix('.kernels.csv').open('x',buffering=1024*1024) as f:
   w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(kernels)
  print(path.name,json.dumps({k:result[k] for k in ['mode','wall_ms','gpu_busy_union_ms','gpu_overlap_ms','unclassified_kernel_sum_ms']}))
