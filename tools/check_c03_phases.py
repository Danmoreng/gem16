#!/usr/bin/env python3
"""C03 deterministic phase/deadline probe on real models, test-only holds."""
import argparse, base64, concurrent.futures, hashlib, http.client, io, json
from pathlib import Path
import signal, socket, struct, subprocess, sys, time, uuid, wave
from hf_cache import default_target_model, default_assistant_model, locked_snapshot_path
ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--server', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--profile', choices=['12b', '26b'], required=True)
    ap.add_argument('--draft', type=int, choices=[0, 2], default=0)
    ap.add_argument('--port', type=int, default=18087)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    command = [str(a.server.resolve()), '--model-name', 'gem16', '--port', str(a.port),
               '--max-context', '32768', '--max-sessions', '2' if a.profile == '12b' else '1',
               '--max-queued-requests', '2', '--greedy', '--log-format', 'json']
    if a.profile == '12b':
        command += ['--model', str(default_target_model())]
        assistant = default_assistant_model()
    else:
        for flag, component in [('--model', 'trellis35-target'), ('--vision-model', 'vision-fp8')]:
            command += [flag, str(locked_snapshot_path(ROOT / f'models/gemma4-26b-{component}.lock.json'))]
        assistant = locked_snapshot_path(ROOT / 'models/gemma4-26b-gem16-assistant.lock.json')
    if a.draft:
        command += ['--assistant-model', str(assistant), '--mtp-draft-tokens', str(a.draft)]
    report = {'command': command, 'cases': [], 'commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'server_sha256': hashlib.file_digest(a.server.open('rb'), 'sha256').hexdigest(),
              'scope': 'Real model; explicit test-only 35s holds prove phase arrival. Latency excludes hold before cancellation. No generation wall-time deadline is imposed.'}
    short = {'model':'gem16','messages':[{'role':'user','content':'Reply with exactly: ready'}],
             'reasoning_effort':'none','max_completion_tokens':32}
    pixels = bytes([64,128,255]) * (1024*1024)
    bmp = b'BM'+struct.pack('<IHHI',54+len(pixels),0,0,54)+struct.pack('<IiiHHIIiiII',40,1024,1024,1,24,0,len(pixels),0,0,0,0)+pixels
    image = {'type':'image_url','image_url':{'url':'data:image/bmp;base64,'+base64.b64encode(bmp).decode()}}
    buffer = io.BytesIO()
    with wave.open(buffer,'wb') as wav:
        wav.setparams((1,2,16000,0,'NONE','not compressed')); wav.writeframes(bytes(32000))
    audio = {'type':'input_audio','input_audio':{'format':'wav','data':base64.b64encode(buffer.getvalue()).decode()}}
    media = {**short,'messages':[{'role':'user','content':[image,{'type':'text','text':'Name the color briefly.'}]}]}
    audio_body = {**short,'messages':[{'role':'user','content':[audio,{'type':'text','text':'Describe the audio briefly.'}]}]}
    def request(path, body=None, headers=None):
        c=http.client.HTTPConnection('127.0.0.1',a.port,timeout=80)
        try:
            c.request('GET' if body is None else 'POST',path,None if body is None else json.dumps(body),{'Content-Type':'application/json',**(headers or {})})
            r=c.getresponse(); data=r.read().decode(); return r.status, data
        finally: c.close()
    def metrics():
        status, text=request('/metrics'); assert status==200
        return {line.split()[0]:float(line.split()[1]) for line in text.splitlines() if line and not line.startswith('#')}
    def wait(predicate, timeout=20):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            m=metrics()
            if predicate(m): return m
            time.sleep(.01)
        raise AssertionError(f'condition not reached: {m}')
    def idle():
        return wait(lambda m:m['gem16_request_queue_active']==0
                    and m['gem16_request_queue_depth']==0
                    and m['gem16_active_requests']==0)
    def raw(body, stage=None, affinity=None, route='/v1/chat/completions'):
        if stage == 'generation':
            body = {**body}
            if route.endswith('responses'):
                body.update(input='Count from 1 to 1000, comma separated.', max_output_tokens=128)
            else:
                body.update(messages=[{'role':'user','content':'Count from 1 to 1000, comma separated.'}], max_completion_tokens=128)
        data=json.dumps(body).encode(); c=socket.create_connection(('127.0.0.1',a.port),timeout=50)
        extra=(f'X-Gem16-Test-Fault: {stage}:wait\r\n' if stage else '')+(f'session_id: {affinity}\r\n' if affinity else '')
        c.sendall((f'POST {route} HTTP/1.1\r\nHost: 127.0.0.1:{a.port}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n{extra}\r\n').encode()+data)
        if stage: wait(lambda m:m['gem16_test_pause_active']==1)
        return c
    def close(c):
        try: c.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        c.close()
    def recovery(body=short):
        status,data=request('/v1/chat/completions',body,{'session_id':uuid.uuid4().hex})
        assert status==200,(status,data)
        return json.loads(data)
    def record(name, **details):
        report['cases'].append({'case':name,**details}); print(name,flush=True)
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if sys.platform == 'win32':
        startup=subprocess.STARTUPINFO(); startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow=0
        launch={'creationflags':subprocess.CREATE_NEW_CONSOLE,'startupinfo':startup}
    else:
        launch={'start_new_session':True}
    log=(a.output/'server.txt').open('w',encoding='utf-8')
    process=None
    def start():
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,**launch)
        end=time.monotonic()+180
        while time.monotonic()<end:
            assert p.poll() is None, p.returncode
            try:
                if request('/health')[0]==200: return p
            except OSError: pass
            time.sleep(.2)
        raise TimeoutError('startup')
    def stop(stage):
        begin=time.monotonic()
        if sys.platform != 'win32':
            shutdown_signal = signal.SIGINT if stage == 'image_resize' else signal.SIGTERM
            process.send_signal(shutdown_signal)
            code=process.wait(timeout=20); assert code==0,code
            return time.monotonic()-begin, shutdown_signal.name
        helper="""import ctypes,sys,time
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.FreeConsole()
assert k.AttachConsole(int(sys.argv[1]))
assert k.SetConsoleCtrlHandler(None,True)
assert k.GenerateConsoleCtrlEvent(0,0)
time.sleep(.2)
k.FreeConsole()
"""
        subprocess.run([sys.executable,'-c',helper,str(process.pid)],check=True)
        code=process.wait(timeout=20); assert code==0,code
        return time.monotonic()-begin, 'CTRL_C_EVENT'
    try:
        process=start()
        recovery()
        # The first emitted token holds the real inference lease. EOS cannot
        # accidentally end the blocker before the production 30s wait deadline.
        affinity=uuid.uuid4().hex
        before_blocker=metrics()
        blocker=raw(short,'generation',affinity)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            def waiter():
                begin=time.monotonic(); status,data=request('/v1/chat/completions',short,{'session_id':affinity})
                return {'status':status,'body':data,'seconds':time.monotonic()-begin}
            pending=[pool.submit(waiter) for _ in range(3 if a.profile=='12b' else 2)]
            wait(lambda m:m['gem16_request_queue_depth']==2)
            begin=time.monotonic(); assert request('/health')[0]==200; control=time.monotonic()-begin
            outcomes=[p.result(timeout=40) for p in pending]
        assert all(x['status']==503 and 29<=x['seconds']<33 for x in outcomes), outcomes
        begin=time.monotonic(); close(blocker); after_blocker=idle(); latency=time.monotonic()-begin
        assert after_blocker['gem16_input_tokens_total']==before_blocker['gem16_input_tokens_total']
        record('queue_and_same_session_deadline',waiters=outcomes,control_seconds=control,recovery_seconds=latency,next_response=recovery())
        # Both protocols and response modes use the same CPU preparation scope.
        stages=['request_prepare','media_base64','image_decode','image_resize','image_patchify']
        if a.profile=='12b': stages+=['audio_decode']
        for stage in stages:
            payload=audio_body if stage=='audio_decode' else media
            for streaming in [False,True]:
                for route in ['/v1/chat/completions','/v1/responses']:
                    body={**payload,'stream':streaming}
                    if route.endswith('responses'):
                        content=[]
                        for item in payload['messages'][0]['content']:
                            if item['type']=='text': content.append({'type':'input_text','text':item['text']})
                            elif item['type']=='image_url': content.append({'type':'input_image','image_url':item['image_url']['url']})
                            else: content.append(item)
                        body={'model':'gem16','input':[{'role':'user','content':content}],'stream':streaming,'reasoning':{'effort':'none'},'max_output_tokens':32}
                    c=raw(body,stage,route=route)
                    begin=time.monotonic(); assert request('/health')[0]==200; control=time.monotonic()-begin
                    begin=time.monotonic(); close(c); after=idle(); elapsed=time.monotonic()-begin
                    assert after['gem16_test_pause_active']==0
                    record('media_disconnect',stage=stage,route=route,stream=streaming,seconds=elapsed,control_seconds=control)
            record('media_recovery',stage=stage,response=recovery(payload))
        affinity=uuid.uuid4().hex
        status,data=request('/v1/chat/completions',media,{'session_id':affinity}); assert status==200,(status,data)
        seed=json.loads(data)
        continuation={**media,'messages':media['messages']+[seed['choices'][0]['message'],{'role':'user','content':'Now reply with exactly: ready'}]}
        before=metrics(); c=raw(continuation,'image_patchify',affinity); close(c); idle()
        after_cancel=metrics()
        assert after_cancel['gem16_sessions_created_total']==before['gem16_sessions_created_total']
        assert after_cancel['gem16_sessions_rebuilt_total']==before['gem16_sessions_rebuilt_total']
        status,data=request('/v1/chat/completions',continuation,{'session_id':affinity}); assert status==200,(status,data)
        after=metrics()
        assert after['gem16_cached_input_tokens_total']>before['gem16_cached_input_tokens_total']
        record('media_history_recovery',before=before,after_cancel=after_cancel,after=after,response=json.loads(data))
        # A real media-preparation request must consume the original admission
        # budget, before session acquisition, rather than starting another timer.
        begin=time.monotonic(); status,data=request('/v1/chat/completions',media,{'X-Gem16-Test-Fault':'image_resize:wait'})
        elapsed=time.monotonic()-begin
        assert status==503 and 29<=elapsed<33,(status,data,elapsed)
        record('media_deadline',status=status,body=data,seconds=elapsed,next_response=recovery(media))
        for streaming in [False,True]:
            for route in ['/v1/chat/completions','/v1/responses']:
                body={**short,'stream':streaming} if route.endswith('completions') else {'model':'gem16','input':'Reply with exactly: ready','reasoning':{'effort':'none'},'max_output_tokens':32,'stream':streaming}
                c=raw(body,'generation',route=route)
                begin=time.monotonic(); close(c); idle(); elapsed=time.monotonic()-begin
                record('decode_disconnect',route=route,stream=streaming,seconds=elapsed,next_response=recovery())
        response_body={'model':'gem16','input':'Reply with exactly: ready','reasoning':{'effort':'none'},'max_output_tokens':32,'stream':True}
        c=raw(response_body,'generation',route='/v1/responses')
        stream=http.client.HTTPResponse(c); stream.begin()
        events=[]
        while line:=stream.readline():
            if line.startswith(b'data: {'):
                event=json.loads(line[6:]); events.append(event)
                if event.get('type')=='response.created': break
        response_id=events[-1]['response']['id']
        begin=time.monotonic(); status,data=request(f'/v1/responses/{response_id}/cancel',{})
        control=time.monotonic()-begin; assert status==200,(status,data)
        while line:=stream.readline():
            if line.startswith(b'data: {'): events.append(json.loads(line[6:]))
        idle(); elapsed=time.monotonic()-begin; stream.close(); close(c)
        assert events[-1]['type']=='error',events
        record('decode_explicit_cancel',seconds=elapsed,control_seconds=control,events=events,next_response=recovery())
        # Shutdown in CPU preparation and decode, with queued/pool waiters.
        for stage,payload in [('image_resize',media),('generation',short)]:
            affinity=uuid.uuid4().hex; c=raw(payload,stage,affinity)
            waiting=[raw(short,affinity=affinity) for _ in range(3 if a.profile=='12b' else 2)] if stage=='generation' else []
            if waiting: wait(lambda m:m['gem16_request_queue_depth']==2)
            elapsed, shutdown_signal=stop(stage)
            for sock in [c,*waiting]: close(sock)
            record('graceful_stop',stage=stage,signal=shutdown_signal,seconds=elapsed,exit_code=0)
            process=start(); record('restart',stage=stage,response=recovery())
        report['passed']=True
    except BaseException as error:
        report['passed']=False; report['error']=repr(error); raise
    finally:
        if process is not None and process.poll() is None: process.terminate(); process.wait(timeout=30)
        log.close(); (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
