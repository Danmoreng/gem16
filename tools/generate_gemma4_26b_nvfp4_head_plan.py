#!/usr/bin/env python3
"""Generate the deterministic M07 provisional NVFP4 tied-head plan/config."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.gem16_compile.common import canonical_json_bytes, tensor_bytes
from tools.gem16_compile.profiles import (M05_SOURCE_LOCK_SHA256, M07_COMPONENT_LAYOUTS,
 M07_DEQUANTIZATION_EQUATION, M07_PROFILE, M07_QUANTIZER_PARAMETERS, M07_SOURCE_CONTRACT,
 m07_component_parameters, classify_m05_source, m06_expected_source_specs)
INVENTORY=ROOT/'benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json'
QAT_LOCK=ROOT/'models/gemma4-26b-qat-bf16.lock.json'
OUTPUT_PLAN=ROOT/'benchmarks/goldens/gemma4_26b/nvfp4/qat-head-compiler-plan.json'
OUTPUT_CONFIG=ROOT/'artifacts/m07/nvfp4-head-compiler-config.json'
NVFP4_SPEC=ROOT/'tools/gem16_compile/specs/nvfp4-experts-v1.json'
OMITTED=['audio','mtp','video','vision']
ENV={'byteorder':'little','locale':'C.UTF-8','machine':'x86_64','python_implementation':'CPython','python_major_minor':'3.14','python_version':'3.14.6','system':'Linux'}
HEAD='model.language_model.embed_tokens.weight'; SHAPE=(262144,2816)
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def records():
 stem=HEAD.removesuffix('.weight'); common={'aliased':True,'axis_transformation':'vocabulary,hidden','dequantization_equation':M07_DEQUANTIZATION_EQUATION,'logical_dtype':'BF16','logical_shape':list(SHAPE),'operation_id':'nvfp4-head:model.language_model.embed_tokens','residency_class':'immutable_device_text','role':'tied_embedding_and_output','source_names':[HEAD],'transformation_version':1}
 result=[]
 for component in ('weight_packed','weight_scale','weight_global_scale','input_global_scale'):
  layout=M07_COMPONENT_LAYOUTS[component]; shape=[262144,1408] if component=='weight_packed' else [262144,176] if component=='weight_scale' else [1]
  r=dict(common); r.update({'disk_layout':layout['disk_layout'],'encoder':layout['encoder'],'output_dtype':layout['output_dtype'],'output_name':f'{stem}.{component}','physical_shape':shape,'quantizer_parameters':m07_component_parameters(component),'runtime_layout':layout['runtime_layout_shared'],'transformation':layout['transformation']}); result.append(r)
 return result
def make_plan():
 inv=load(INVENTORY); tensors=inv.get('tensors'); specs=m06_expected_source_specs()
 if inv.get('source_family')!='qat_bf16' or not isinstance(tensors,list) or len(tensors)!=1013 or {x.get('name') for x in tensors if isinstance(x,dict)}!=set(specs): raise ValueError('M07 requires complete 1,013-name QAT inventory')
 by={x['name']:x for x in tensors}; head=by[HEAD]
 if head.get('dtype')!='BF16' or tuple(head.get('shape',()))!=SHAPE or head.get('bytes')!=tensor_bytes('BF16',SHAPE,HEAD): raise ValueError('invalid tied-head source descriptor')
 excluded=[]
 for x in sorted(tensors,key=lambda x:x['name']):
  name=x['name']; role,shape=specs[name]
  if x.get('dtype')!='BF16' or tuple(x.get('shape',()))!=shape or x.get('bytes')!=tensor_bytes('BF16',shape,name): raise ValueError(f'invalid frozen source descriptor: {name}')
  if name==HEAD: continue
  role=classify_m05_source(name); vision=role.startswith('vision_')
  excluded.append({'family':'vision' if vision else 'deferred_non_head','reason':'text-only Gemma 4 26B profile excludes vision tensors' if vision else 'deferred to M08; absent from M07 tied-head partial artifact','residency_class':'compile_excluded_vision' if vision else 'm07_deferred_non_head','role':role,'source_name':x['name']})
 if len(excluded)!=1012: raise ValueError(f'M07 exclusions mismatch: {len(excluded)}')
 return {'approved_metadata_files':[],'artifact_profile':M07_PROFILE.name,'excluded_tensors':excluded,'head_format':'nvfp4','omitted_families':OMITTED,'reference_environment':ENV,'schema_version':1,'source_contract':M07_SOURCE_CONTRACT,'source_lock_sha256':sha(QAT_LOCK),'target_shard_bytes':1<<30,'tensors':sorted(records(),key=lambda x:x['output_name'])}
def lock_ref(path):
 d=load(path); return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'repository':str(d['repository']),'revision':str(d['revision'])}
def make_config(plan):
 inv=load(INVENTORY)
 source_count=len(inv['tensors'])
 tied_sources={name for tensor in plan['tensors'] for name in tensor['source_names']}
 output_bytes=sum(tensor_bytes(item['output_dtype'], tuple(item['physical_shape']), item['output_name']) for item in plan['tensors'])
 counts={
  'source_tensor_count':source_count,
  'tied_source_tensor_count':len(tied_sources),
  'output_tensor_count':len(plan['tensors']),
  'excluded_tensor_count':len(plan['excluded_tensors']),
  'output_tensor_bytes':output_bytes,
  'one_physical_tied_payload':len(tied_sources)==1 and len({item['operation_id'] for item in plan['tensors']})==1,
 }
 if counts['source_tensor_count'] != counts['excluded_tensor_count'] + counts['tied_source_tensor_count'] or output_bytes != 415236104 or not counts['one_physical_tied_payload']:
  raise ValueError('M07 derived counts do not satisfy the tied-head contract')
 spec={'schema_version':1,'milestone':'M07','artifact_profile':M07_PROFILE.name,'source_contract':M07_SOURCE_CONTRACT,'source':{'qat_bf16':lock_ref(QAT_LOCK)},'quantizer':{'spec_path':'tools/gem16_compile/specs/nvfp4-experts-v1.json','spec_sha256':sha(NVFP4_SPEC),'contract':dict(M07_QUANTIZER_PARAMETERS)},'counts':counts,'reference_semantics':{'vocabulary_size':262144,'hidden_size':2816,'embedding_scale':'BF16-RNE(sqrt(hidden_size))','final_logit_softcap':30.0,'tie_break':'lowest_token_id','suppression':'caller_supplied_strict_ids_max_16','lookup':'decode canonical NVFP4 row then BF16-RNE boundary','projection':'post-final-normalization BF16-RNE hidden; quantize activation with stored input divisor then dequantized dot'},'diagnostics':{'status':'not_run','quality_claim':False,'runtime_loadable':False}}
 return spec
def generate():
 plan=make_plan(); return {OUTPUT_PLAN:canonical_json_bytes(plan),OUTPUT_CONFIG:canonical_json_bytes(make_config(plan))}
def main(argv=None):
 ap=argparse.ArgumentParser(); ap.add_argument('--check',action='store_true'); a=ap.parse_args(argv); out=generate(); bad=[]
 for p,b in out.items():
  if a.check:
   try:
    if p.read_bytes()!=b: bad.append(str(p))
   except OSError: bad.append(str(p))
  else: p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)
 if bad: raise SystemExit('generated outputs differ: '+', '.join(bad))
 return 0
if __name__=='__main__': raise SystemExit(main())
