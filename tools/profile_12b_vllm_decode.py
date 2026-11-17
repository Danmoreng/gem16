#!/usr/bin/env python3
"""Capture a bounded warm vLLM decode window; never primary throughput evidence."""
from pathlib import Path
import functools
import json
import os
import sys


def main():
    import vllm
    import benchmark_dflash_workload as runner
    import benchmark_wikipedia_workload as core

    traces = Path(os.environ['GEM16_VLLM_TRACE_DIR']).resolve()
    traces.mkdir(parents=True, exist_ok=False)
    delay = int(os.environ.get('GEM16_VLLM_TRACE_DELAY', '32'))
    limit = int(os.environ.get('GEM16_VLLM_TRACE_STEPS', '16'))
    vllm.LLM = functools.partial(vllm.LLM, profiler_config={
        'profiler': 'torch', 'torch_profiler_dir': str(traces),
        'ignore_frontend': True, 'torch_profiler_with_stack': False,
        'torch_profiler_record_shapes': True, 'torch_profiler_with_memory': False,
        'detailed_trace_annotation': True})
    original = core.run_vllm_request
    calls = 0

    def traced(llm, *args, **kwargs):
        nonlocal calls
        collect = calls == 1
        calls += 1
        if collect:
            llm.start_profile(profile_prefix='12b-steady-decode',
                              delay_iterations=delay, max_iterations=limit)
        try:
            return original(llm, *args, **kwargs)
        finally:
            if collect:
                llm.stop_profile()

    core.run_vllm_request = traced
    runner.main()
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    doc = json.loads(output.read_text())
    doc['status'] = 'diagnostic_only_profiled_not_primary_throughput'
    doc['profile'] = {'trace_dir': str(traces), 'delay_worker_iterations': delay,
                      'active_worker_iterations': limit,
                      'scope_must_be_verified_from_trace': True,
                      'wrapper': str(Path(__file__).resolve())}
    output.write_text(json.dumps(doc, indent=2) + '\n')


if __name__ == '__main__':
    main()
