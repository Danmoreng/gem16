#!/usr/bin/env python3
"""Bounded Ordinary12B worker trace; profiled rates are diagnostic only."""
from pathlib import Path
import functools
import json
import os
import sys


def main():
    import vllm
    import benchmark_wikipedia_workload as core

    if '--engine' not in sys.argv or sys.argv[sys.argv.index('--engine')+1] != 'vllm':
        raise SystemExit('this diagnostic requires --engine vllm')
    if '--mtp-draft-tokens' in sys.argv and sys.argv[sys.argv.index('--mtp-draft-tokens')+1] != '0':
        raise SystemExit('Ordinary diagnostic requires zero speculative tokens')
    traces=Path(os.environ['GEM16_VLLM_TRACE_DIR']).resolve()
    traces.mkdir(parents=True,exist_ok=False)
    delay=int(os.environ.get('GEM16_VLLM_TRACE_DELAY','48'))
    limit=int(os.environ.get('GEM16_VLLM_TRACE_STEPS','16'))
    vllm.LLM=functools.partial(vllm.LLM,profiler_config={
        'profiler':'torch','torch_profiler_dir':str(traces),'ignore_frontend':True,
        'torch_profiler_with_stack':False,'torch_profiler_record_shapes':True,
        'torch_profiler_with_memory':False,'detailed_trace_annotation':True})
    original=core.run_vllm_request
    calls=0
    def traced(llm,*args,**kwargs):
        nonlocal calls
        collect=calls==1;calls+=1
        if collect:llm.start_profile(profile_prefix='12b-ordinary',delay_iterations=delay,max_iterations=limit)
        try:return original(llm,*args,**kwargs)
        finally:
            if collect:llm.stop_profile()
    core.run_vllm_request=traced
    core.main()
    output=Path(sys.argv[sys.argv.index('--output')+1]);d=json.loads(output.read_text())
    d['status']='diagnostic_only_profiled_not_primary_throughput'
    d['profile']={'trace_dir':str(traces),'delay_worker_iterations':delay,
                  'active_worker_iterations':limit,'scope_must_be_verified_from_trace':True,
                  'wrapper':str(Path(__file__).resolve())}
    output.write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':main()
