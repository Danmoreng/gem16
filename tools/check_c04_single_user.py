#!/usr/bin/env python3
"""C04: eight sequential image requests, global ten-minute deadline."""
import argparse
import ctypes
import hashlib
import http.client
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import uuid

from check_multi_image_conversation import image_part
from hf_cache import default_target_model, locked_snapshot_path

ROOT = Path(__file__).resolve().parents[1]

if sys.platform == 'win32':
    from ctypes import wintypes

    class ProcessMemory(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18088)
    parser.add_argument('--resume-result', type=Path, help='Continue after a failed first request without repeating it or resetting the global budget')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    started = time.monotonic()
    previous = None
    elapsed_before = 0
    if args.resume_result:
        previous = json.loads(args.resume_result.read_text(encoding='utf-8'))
        assert previous['request_count'] == 1 and len(previous['profiles']) == 1
        assert previous['profiles'][0]['cases'][0]['case'] == 'one_image'
        origin = args.resume_result.stat().st_mtime - previous['elapsed_seconds']
        elapsed_before = time.time()-origin
    deadline = started + 600 - elapsed_before
    windows = sys.platform == 'win32'
    report = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'server_sha256': hashlib.sha256(args.server.read_bytes()).hexdigest(),
              'scope': f'{sys.platform}, one slot, ordinary decode, eight requests maximum, 600 seconds including model startup',
              'memory_method': ('GetProcessMemoryInfo every 10ms: sampled working set and private commit; OS lifetime PeakWorkingSetSize includes startup'
                                if windows else '/proc status and smaps_rollup every 10ms: sampled RSS, private RSS and VmHWM include startup'),
              'profiles': [], 'request_count': previous['request_count'] if previous else 0,
              'elapsed_before_resume_seconds': elapsed_before, 'prior_result': str(args.resume_result) if previous else None}
    if windows:
        query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
        query.restype = wintypes.BOOL

    def query_memory(process):
        if windows:
            memory = ProcessMemory()
            memory.cb = ctypes.sizeof(memory)
            if not query(wintypes.HANDLE(int(process._handle)), ctypes.byref(memory), memory.cb):
                return None
            return {'working_set': memory.WorkingSetSize,
                    'private_commit': memory.PrivateUsage,
                    'lifetime_peak_working_set': memory.PeakWorkingSetSize}
        try:
            status = {}
            for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    key, value, _ = line.split()
                    status[key[:-1]] = int(value) * 1024
            private_rss = 0
            for line in Path(f'/proc/{process.pid}/smaps_rollup').read_text().splitlines():
                if line.startswith(('Private_Clean:', 'Private_Dirty:')):
                    private_rss += int(line.split()[1]) * 1024
            return {'working_set': status['VmRSS'], 'private_commit': private_rss,
                    'lifetime_peak_working_set': status['VmHWM']}
        except (FileNotFoundError, KeyError, PermissionError, ProcessLookupError, ValueError):
            return None
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    names = ['red', 'green', 'blue', 'yellow']
    images = [image_part(color) for color in colors]
    report['fixture_sha256'] = [hashlib.sha256(json.dumps(image, sort_keys=True).encode()).hexdigest() for image in images]

    def request(route, body=None, session=None):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('global ten-minute limit')
        connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=min(60, remaining))
        try:
            headers = {'Content-Type': 'application/json'}
            if session:
                headers['X-Gem16-Session-Id'] = session
            connection.request('GET' if body is None else 'POST', route,
                               None if body is None else json.dumps(body), headers)
            response = connection.getresponse()
            return response.status, response.read().decode(), response.getheader('X-Gem16-Session-Id')
        finally:
            connection.close()

    def metrics():
        status, data, _ = request('/metrics')
        assert status == 200
        return {line.split()[0]: float(line.split()[1]) for line in data.splitlines()
                if line and not line.startswith('#')}

    try:
        for profile in ['12b', '26b']:
            command = [str(args.server.resolve()), '--model-name', 'gem16', '--port', str(args.port),
                       '--max-context', '32768', '--max-sessions', '1', '--max-queued-requests', '1',
                       '--greedy', '--log-format', 'json']
            if profile == '12b':
                command += ['--model', str(default_target_model())]
            else:
                for flag, component in [('--model', 'trellis35-target'), ('--vision-model', 'vision-fp8')]:
                    command += [flag, str(locked_snapshot_path(ROOT / f'models/gemma4-26b-{component}.lock.json'))]
            item = {'profile': profile, 'command': command, 'cases': list(previous['profiles'][0]['cases']) if previous and profile == '12b' else []}
            report['profiles'].append(item)
            samples = []
            phase = 'startup'
            stopping = threading.Event()
            with (args.output/f'{profile}-server.txt').open('w', encoding='utf-8') as log:
                launch = {'creationflags': subprocess.CREATE_NO_WINDOW} if windows else {'start_new_session': True}
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **launch)
                def sample():
                    while not stopping.is_set():
                        if time.monotonic() >= deadline:
                            process.terminate()
                            return
                        memory = query_memory(process)
                        if memory:
                            samples.append({'seconds': time.monotonic()-started, 'phase': phase,
                                            **memory})
                        stopping.wait(.01)
                sampler = threading.Thread(target=sample, daemon=True)
                sampler.start()
                try:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f'{profile} server exited {process.returncode}')
                        try:
                            if request('/health')[0] == 200:
                                break
                        except OSError:
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError('global ten-minute limit')
                        time.sleep(.1)
                    phase = 'ready'
                    time.sleep(.05)
                    item['ready_memory'] = samples[-1]
                    messages = []
                    session = None
                    for label, count in [('one_image', 1), ('two_images', 2), ('four_images', 4), ('continue_history', 4)]:
                        if previous and profile == '12b' and label == 'one_image':
                            continue
                        if label != 'continue_history':
                            session = uuid.uuid4().hex
                            messages = [{'role': 'user', 'content': [*images[:count], {'type': 'text',
                                'text': 'List the colors of these images in their given order. Use only comma-separated English color names.'}]}]
                        else:
                            messages.append({'role': 'user', 'content': 'List the colors of all four previous images again in the same order. Only comma-separated English color names.'})
                        body = {'model': 'gem16', 'messages': messages, 'reasoning_effort': 'none', 'max_completion_tokens': 64}
                        before = metrics()
                        assert report['request_count'] < 8
                        report['request_count'] += 1
                        phase = label
                        begin = time.monotonic()
                        status, data, returned_session = request('/v1/chat/completions', body, session)
                        elapsed = time.monotonic()-begin
                        time.sleep(.03)
                        phase = 'between_requests'
                        after = metrics()
                        response = json.loads(data)
                        entry = {'case': label, 'image_count_in_history': count, 'status': status,
                                 'seconds': elapsed, 'session': returned_session, 'response': response,
                                 'metrics_before': before, 'metrics_after': after}
                        item['cases'].append(entry)
                        assert status == 200, response
                        assert returned_session == session, 'session identity changed'
                        message = response['choices'][0]['message']
                        recalled = re.findall(r'\b(red|green|blue|yellow)\b', message.get('content', '').lower())
                        entry['color_order_passed'] = recalled == names[:count]
                        # Semantic failures are retained while finishing the fixed eight-case scope.
                        assert after['gem16_resident_sessions'] == 1
                        if label == 'continue_history':
                            cached = response['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
                            entry['cached_tokens'] = cached
                            assert cached > 0
                            assert after['gem16_sessions_created_total'] == before['gem16_sessions_created_total']
                            assert after['gem16_sessions_rebuilt_total'] == before['gem16_sessions_rebuilt_total']
                        messages.append({k: v for k, v in message.items() if k in ('role', 'content', 'tool_calls')})
                        print(profile, label, message.get('content'), round(elapsed, 3), flush=True)
                    item['passed'] = all(case.get('color_order_passed', False) for case in item['cases'])
                finally:
                    stopping.set()
                    sampler.join(timeout=2)
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
                    (args.output/f'{profile}-memory.json').write_text(json.dumps(samples, indent=2)+'\n')
                    for case in item['cases']:
                        rows = [row for row in samples if row['phase'] == case['case']]
                        if rows:
                            case['sampled_peak_working_set'] = max(row['working_set'] for row in rows)
                            case['sampled_peak_private_commit'] = max(row['private_commit'] for row in rows)
                            case['post_request_working_set'] = rows[-1]['working_set']
                    if samples:
                        item['lifetime_peak_working_set_including_startup'] = max(row['lifetime_peak_working_set'] for row in samples)
        report['completed_scope'] = report['request_count'] == 8
        report['passed'] = report['completed_scope'] and all(item['passed'] for item in report['profiles'])
    except BaseException as error:
        report['passed'] = False
        report['error'] = repr(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started + elapsed_before
        report['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
