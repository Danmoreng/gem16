#!/usr/bin/env python3
"""Bounded live pre-output cancellation/recovery probe; no release qualification claim."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

from hf_cache import default_target_model, default_assistant_model, locked_snapshot_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=['12b', '26b'], required=True)
    parser.add_argument('--draft', type=int, choices=[0, 2], default=0)
    parser.add_argument('--port', type=int, default=18086)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as check:
        check.bind(('127.0.0.1', args.port))
    command = [str(args.server.resolve()), '--model-name', 'gem16', '--port', str(args.port),
               '--max-context', '32768', '--max-sessions', '2' if args.profile == '12b' else '1',
               '--max-queued-requests', '2', '--greedy', '--log-format', 'json']
    if args.profile == '12b':
        command += ['--model', str(default_target_model())]
        assistant = default_assistant_model()
    else:
        for flag, component in [('--model', 'trellis35-target'), ('--vision-model', 'vision-fp8')]:
            command += [flag, str(locked_snapshot_path(ROOT / f'models/gemma4-26b-{component}.lock.json'))]
        assistant = locked_snapshot_path(ROOT / 'models/gemma4-26b-gem16-assistant.lock.json')
    if args.draft:
        command += ['--assistant-model', str(assistant), '--mtp-draft-tokens', str(args.draft)]
    report = {'command': command, 'cases': [], 'scope': '32K capacity, pre-output cancellation; not everyday-context qualification',
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'server_sha256': hashlib.file_digest(args.server.open('rb'), 'sha256').hexdigest()}

    def request(path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', args.port, timeout=120)
        try:
            conn.request('GET' if body is None else 'POST', path,
                         None if body is None else json.dumps(body),
                         {'Content-Type': 'application/json', **(headers or {})})
            response = conn.getresponse()
            data = response.read().decode()
            return response.status, json.loads(data) if path != '/metrics' else data
        finally:
            conn.close()

    def metrics():
        status, data = request('/metrics')
        assert status == 200
        return {line.split()[0]: float(line.split()[1]) for line in data.splitlines()
                if line and not line.startswith('#')}

    def wait_active(value, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = metrics()
            if current['gem16_active_requests'] == value:
                return current
            time.sleep(.02)
        raise AssertionError(f'active requests did not reach {value}')

    prompt = 'Remember this list of words:\n' + ' apple orange banana pear' * 3000 + '\nReply with exactly: ready'
    body = {'model': 'gem16', 'messages': [{'role': 'user', 'content': prompt}],
            'reasoning_effort': 'none', 'max_completion_tokens': 32}
    short = {**body, 'messages': [{'role': 'user', 'content': 'Reply with exactly: ready'}]}
    with (args.output / 'server.txt').open('w', encoding='utf-8') as log:
        launch = {}
        if args.extended and sys.platform == 'win32':
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            launch = {'creationflags': subprocess.CREATE_NEW_CONSOLE, 'startupinfo': startup}
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **launch)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited {process.returncode}')
                try:
                    if request('/health')[0] == 200:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('startup')
                time.sleep(.25)
            for index in range(3):
                before = metrics()
                begin = time.monotonic()
                status, response = request('/v1/chat/completions', body, {'session_id': uuid.uuid4().hex})
                after = metrics()
                assert status == 200, response
                assert response['usage']['prompt_tokens'] > 8000
                report['cases'].append({'case': 'uncancelled', 'sample': index, 'seconds': time.monotonic()-begin,
                                        'response': response, 'prompt_us': after['gem16_prompt_microseconds_total']-before['gem16_prompt_microseconds_total']})
            if not args.baseline_only:
                for route in ['/v1/chat/completions', '/v1/responses']:
                    for streaming in [False, True]:
                        payload = {**body, 'stream': streaming}
                        if route.endswith('responses'):
                            payload = {'model': 'gem16', 'input': prompt, 'max_output_tokens': 32,
                                       'reasoning': {'effort': 'none'}, 'stream': streaming}
                        raw = json.dumps(payload).encode()
                        before = metrics()
                        sock = socket.create_connection(('127.0.0.1', args.port), timeout=10)
                        sock.sendall((f'POST {route} HTTP/1.1\r\nHost: 127.0.0.1:{args.port}\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n').encode()+raw)
                        wait_active(1)
                        time.sleep(.4)
                        # Health must remain responsive while GPU prefill is active.
                        begin = time.monotonic()
                        assert request('/health')[0] == 200
                        health_seconds = time.monotonic()-begin
                        begin = time.monotonic()
                        sock.shutdown(socket.SHUT_RDWR)
                        sock.close()
                        after = wait_active(0)
                        elapsed = time.monotonic()-begin
                        assert after['gem16_client_disconnects_total'] > before['gem16_client_disconnects_total']
                        status, recovery = request('/v1/chat/completions', short)
                        assert status == 200, recovery
                        report['cases'].append({'case': 'disconnect', 'route': route, 'stream': streaming,
                                                'recovery_seconds': elapsed, 'health_seconds': health_seconds,
                                                'before': before, 'after': after, 'next_response': recovery})
                        print(route, streaming, elapsed, flush=True)
                if args.extended:
                    # Occupy admission with one active session and a same-session
                    # waiter (12B has two slots), then fill the bounded FIFO.
                    waiting_sockets = []
                    affinity = uuid.uuid4().hex
                    before = metrics()
                    try:
                        for index in range(5 if args.profile == '12b' else 4):
                            raw = json.dumps(body).encode()
                            waiting = socket.create_connection(('127.0.0.1', args.port), timeout=10)
                            waiting_sockets.append(waiting)
                            waiting.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{args.port}\r\nsession_id: {affinity}\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n').encode()+raw)
                            if index == 0:
                                wait_active(1)
                            time.sleep(.03)
                        final = waiting_sockets.pop()
                        with final:
                            assert b' 503 ' in final.recv(4096).split(b'\r\n')[0]
                        begin = time.monotonic()
                        assert request('/health')[0] == 200
                        control_seconds = time.monotonic()-begin
                    finally:
                        for waiting in reversed(waiting_sockets):
                            waiting.shutdown(socket.SHUT_RDWR)
                            waiting.close()
                    begin = time.monotonic()
                    after = wait_active(0)
                    elapsed = time.monotonic()-begin
                    time.sleep(.1)
                    assert metrics()['gem16_input_tokens_total'] == before['gem16_input_tokens_total']
                    assert request('/v1/chat/completions', short)[0] == 200
                    report['cases'].append({'case': 'saturated_same_session_disconnects',
                                            'control_seconds': control_seconds, 'recovery_seconds': elapsed,
                                            'before': before, 'after': after})
                    # Keep reading the SSE stream so cancellation is explicit,
                    # not a side effect of closing the transport.
                    created = threading.Event()
                    events = []
                    def stream_response():
                        conn = http.client.HTTPConnection('127.0.0.1', args.port, timeout=30)
                        try:
                            conn.request('POST', '/v1/responses', json.dumps({
                                'model': 'gem16', 'input': prompt, 'max_output_tokens': 32,
                                'reasoning': {'effort': 'none'}, 'stream': True}),
                                {'Content-Type': 'application/json'})
                            response = conn.getresponse()
                            assert response.status == 200
                            while line := response.readline():
                                if line.startswith(b'data: {'):
                                    event = json.loads(line[6:])
                                    events.append(event)
                                    if event.get('type') == 'response.created':
                                        created.set()
                        finally:
                            conn.close()
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        reading = pool.submit(stream_response)
                        assert created.wait(10), 'missing response.created'
                        time.sleep(.4)
                        response_id = events[0]['response']['id']
                        begin = time.monotonic()
                        status, cancelled = request(f'/v1/responses/{response_id}/cancel', {})
                        assert status == 200, cancelled
                        control_seconds = time.monotonic()-begin
                        reading.result(timeout=15)
                        elapsed = time.monotonic()-begin
                    assert not any(e.get('type') == 'response.output_text.delta' for e in events)
                    assert events[-1].get('type') == 'error'
                    assert wait_active(0)['gem16_active_requests'] == 0
                    assert request('/v1/chat/completions', short)[0] == 200
                    report['cases'].append({'case': 'explicit_cancel', 'control_seconds': control_seconds,
                                            'recovery_seconds': elapsed, 'events': events})
                    if sys.platform == 'win32':
                        raw = json.dumps(body).encode()
                        sock = socket.create_connection(('127.0.0.1', args.port), timeout=10)
                        sock.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{args.port}\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n').encode()+raw)
                        wait_active(1)
                        time.sleep(.4)
                        # Attach only to this test server's hidden console.
                        helper = '''import ctypes,sys,time
k=ctypes.WinDLL('kernel32', use_last_error=True)
k.FreeConsole()
assert k.AttachConsole(int(sys.argv[1]))
assert k.SetConsoleCtrlHandler(None,True)
assert k.GenerateConsoleCtrlEvent(0,0)
time.sleep(.2)
k.FreeConsole()
'''
                        begin = time.monotonic()
                        subprocess.run([sys.executable, '-c', helper, str(process.pid)], check=True)
                        code = process.wait(timeout=15)
                        elapsed = time.monotonic()-begin
                        sock.close()
                        assert code == 0, code
                        assert 'shutdown_completed' in (args.output / 'server.txt').read_text()
                        report['cases'].append({'case': 'windows_ctrl_c_during_prefill', 'seconds': elapsed, 'exit_code': code})
                        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **launch)
                        deadline = time.monotonic()+180
                        while True:
                            try:
                                if request('/health')[0] == 200:
                                    break
                            except OSError:
                                pass
                            if time.monotonic() > deadline or process.poll() is not None:
                                raise RuntimeError('restart failed')
                            time.sleep(.25)
                        assert request('/v1/chat/completions', short)[0] == 200
                        report['cases'].append({'case': 'restart_and_generate', 'passed': True})
            report['passed'] = True
        except BaseException as error:
            report['passed'] = False
            report['error'] = repr(error)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            # terminate is harness cleanup, explicitly NOT graceful-stop evidence.
            (args.output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
