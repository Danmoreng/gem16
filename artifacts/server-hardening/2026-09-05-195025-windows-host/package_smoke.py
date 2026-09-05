import pathlib,json,zipfile,hashlib,subprocess,tempfile
root=pathlib.Path.cwd();archive=root/'build/packages/windows-491f5f5-20260905/gem16-server-0.2.0-dev-windows-x64.zip';out=root/'artifacts/server-hardening/2026-09-05-195025-windows-host'
with zipfile.ZipFile(archive) as z:
 prefix='gem16-server-0.2.0-dev-windows-x64/'
 manifest=json.loads(z.read(prefix+'manifest.json'))
 for name,digest in manifest['files'].items():assert hashlib.sha256(z.read(prefix+name)).hexdigest()==digest,name
 with tempfile.TemporaryDirectory(prefix='gem16-package-check-') as tmp:
  for member in z.infolist():
   target=pathlib.Path(tmp)/member.filename
   assert target.resolve().is_relative_to(pathlib.Path(tmp).resolve())
  z.extractall(tmp)
  result=subprocess.check_output([str(pathlib.Path(tmp)/prefix/'bin/gem16-server.exe'),'--version'],text=True).strip()
  assert result=='gem16-server 0.2.0-dev'
report={'status':'passed','scope':'Same-machine extracted archive integrity/version smoke, not clean-machine or full inference qualification','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest':manifest,'version_output':result}
(out/'package-smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Package manifest hashes and extracted Windows executable version passed')
