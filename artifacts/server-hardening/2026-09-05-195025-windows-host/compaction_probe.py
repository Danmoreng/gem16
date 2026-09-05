import json,pathlib,subprocess,sys,time,urllib.request,hashlib,os
root=pathlib.Path.cwd();baseline=root/'artifacts/server-hardening/2026-09-05-195025-windows-baseline';out=root/'artifacts/server-hardening/2026-09-05-195025-windows-compaction';out.mkdir(exist_ok=False)
report={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'scope':'Windows manual Pi compaction and new-session probe; not fork/resume or automatic compaction','profiles':[]}
for row in json.loads((baseline/'result.json').read_text())['profiles']:
 pdir=out/row['profile'];pdir.mkdir();cmd=row['server_command'];cmd[cmd.index('--port')+1]='18084'
 result={'profile':row['profile'],'server_command':cmd}
 with (pdir/'server.txt').open('w',encoding='utf-8') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  try:
   deadline=time.monotonic()+180
   while True:
    if proc.poll() is not None:raise RuntimeError('server exited')
    try:
     with urllib.request.urlopen('http://127.0.0.1:18084/health',timeout=2) as r:result['health']=json.load(r)
     break
    except OSError:
     if time.monotonic()>deadline:raise
     time.sleep(.25)
   check=[sys.executable,'tools/check_pi_compaction.py','--base-url','http://127.0.0.1:18084','--models-file',str(baseline/row['profile']/'pi/models.json'),'--output-dir',str(pdir/'pi')]
   result['check_command']=check
   with (pdir/'stdout.txt').open('w',encoding='utf-8') as stdout,(pdir/'stderr.txt').open('w',encoding='utf-8') as stderr:
    run=subprocess.run(check,stdout=stdout,stderr=stderr,timeout=480)
   result['exit_code']=run.returncode
  except Exception as e:result['error']=str(e)
  finally:proc.terminate();proc.wait(timeout=30)
 report['profiles'].append(result);print(row['profile'],result.get('exit_code',result.get('error')),flush=True)
report['status']='passed' if all(r.get('exit_code')==0 for r in report['profiles']) else 'failed'
(out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
sys.exit(report['status']!='passed')
