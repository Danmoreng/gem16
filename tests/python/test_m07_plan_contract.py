"""M07 provisional tied-head compiler contract tests."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tools.gem16_compile.common import InvalidPlanError
from tools.gem16_compile.plan import load_quantization_plan
from tools.gem16_compile.profiles import M05_SOURCE_LOCK_SHA256, M07_PROFILE
from tools.gem16_compile.reader import TensorDescriptor, VerifiedSource
from tools.generate_gemma4_26b_nvfp4_head_plan import make_plan
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'benchmarks/goldens/gemma4_26b/nvfp4/qat-head-compiler-plan.json'
class M07PlanContractTest(unittest.TestCase):
 def descriptors(self):
  inv=json.loads((ROOT/'benchmarks/goldens/gemma4_26b/source-inventories/qat-bf16.json').read_text())
  return {x['name']:TensorDescriptor(x['name'],x['dtype'],tuple(x['shape']),'x',Path('/tmp/x'),0,0,x['bytes'],'0'*64) for x in inv['tensors']}
 def source(self,lock=M05_SOURCE_LOCK_SHA256['qat_bf16']):
  return VerifiedSource(Path('/tmp'),Path('/tmp/l'),lock,'repo','0'*40,'now',{})
 def load(self,doc=None,lock=None,descriptors=None):
  with tempfile.TemporaryDirectory(prefix='gem16-m07-plan-') as directory:
   p=Path(directory)/'plan.json';p.write_text(json.dumps(doc or json.loads(PLAN.read_text())))
   return load_quantization_plan(p,self.source(lock or M05_SOURCE_LOCK_SHA256['qat_bf16']),descriptors or self.descriptors(),M07_PROFILE.name,M07_PROFILE.head_format)
 def test_generated_counts_shapes_aliases_and_bytes(self):
  d=make_plan();self.assertEqual(len(d['tensors']),4);self.assertEqual(len(d['excluded_tensors']),1012)
  self.assertTrue(all(x['aliased'] for x in d['tensors']));self.assertEqual(sum({'U8':369098752,'F8_E4M3':46137344,'F32':4}[x['output_dtype']] for x in d['tensors']),415236104)
  self.assertEqual({x['source_names'][0] for x in d['tensors']},{'model.language_model.embed_tokens.weight'})
 def test_wrong_lock_and_lm_head_are_rejected(self):
  with self.assertRaises(InvalidPlanError):self.load(lock='0'*64)
  d=json.loads(PLAN.read_text());d['tensors'][0]['output_name']='lm_head.weight_packed'
  with self.assertRaises(InvalidPlanError):self.load(d)
 def test_wrong_axis_source_and_unknown_keys_are_rejected(self):
  d=json.loads(PLAN.read_text());d['tensors'][0]['axis_transformation']='output,input'
  with self.assertRaises(InvalidPlanError):self.load(d)
  d=json.loads(PLAN.read_text());d['tensors'][0]['source_names']=['other.weight']
  with self.assertRaises(InvalidPlanError):self.load(d)
  d=json.loads(PLAN.read_text());d['tensors'][0]['unexpected']=True
  with self.assertRaises(InvalidPlanError):self.load(d)
  d=json.loads(PLAN.read_text());d['excluded_tensors'][0]['unexpected']=True
  with self.assertRaises(InvalidPlanError):self.load(d)
 def test_every_frozen_source_descriptor_is_validated(self):
  descriptors=self.descriptors()
  name='model.language_model.layers.0.input_layernorm.weight'
  original=descriptors[name]
  for dtype,shape,byte_length in [('F16',original.shape,original.byte_length),(original.dtype,(1,),original.byte_length),(original.dtype,original.shape,2)]:
   mutated=dict(descriptors)
   mutated[name]=TensorDescriptor(name,dtype,shape,original.shard,original.path,original.absolute_offset,original.data_offset,byte_length,original.shard_sha256)
   with self.assertRaises(InvalidPlanError):self.load(descriptors=mutated)
 def test_generator_and_schema_contract_are_current(self):
  result=subprocess.run([sys.executable,str(ROOT/'tools/generate_gemma4_26b_nvfp4_head_plan.py'),'--check'],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr)
  plan_schema=json.loads((ROOT/'tools/gem16_compile/schemas/compiler-plan.schema.json').read_text())
  compilation_schema=json.loads((ROOT/'tools/gem16_compile/schemas/gem16-compilation.schema.json').read_text())
  self.assertIn('nvfp4-tied-head-partial-v1',plan_schema['properties']['artifact_profile']['enum'])
  self.assertIn('nvfp4',plan_schema['properties']['head_format']['enum'])
  self.assertEqual(compilation_schema['properties']['head_format']['enum'],['source','deferred','nvfp4'])
  self.assertIn('compilerM07',compilation_schema['$defs'])
  self.assertEqual(compilation_schema['$defs']['compilerM07']['allOf'][1]['properties']['implementation']['const'],'gem16_compile_m07_native_v1')
  m06_branch=compilation_schema['allOf'][0]['oneOf'][2]
  m07_branch=compilation_schema['allOf'][0]['oneOf'][3]
  self.assertEqual(m06_branch['properties']['tensors']['items']['properties']['aliased']['const'],False)
  self.assertEqual(m07_branch['properties']['tensors']['items']['properties']['aliased']['const'],True)
 def test_m06_and_m07_contract_tables_are_independent(self):
  from tools.gem16_compile.profiles import M06_COMPONENT_LAYOUTS,M06_QUANTIZER_PARAMETERS,M07_COMPONENT_LAYOUTS,M07_QUANTIZER_PARAMETERS
  self.assertIsNot(M06_COMPONENT_LAYOUTS,M07_COMPONENT_LAYOUTS)
  self.assertIsNot(M06_COMPONENT_LAYOUTS['weight_packed'],M07_COMPONENT_LAYOUTS['weight_packed'])
  self.assertIsNot(M06_QUANTIZER_PARAMETERS,M07_QUANTIZER_PARAMETERS)
if __name__=='__main__':unittest.main()
