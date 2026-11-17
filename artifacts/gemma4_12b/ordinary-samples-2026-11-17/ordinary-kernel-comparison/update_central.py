from pathlib import Path
import subprocess,json,tempfile,os,datetime,hashlib,shutil,fcntl
R=Path.cwd();O=Path(__file__).resolve().parent;base=Path('/home/sebastian/Development/_engines/llama.cpp');remote='https://github.com/ggml-org/llama.cpp.git'
lock=(base/'.update.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def run(label,c,timeout=1800):
 (O/(label+'.command.json')).write_text(json.dumps(c,indent=2)+'\n');print('RUN',label,flush=True)
 with (O/(label+'.stdout.log')).open('x') as out,(O/(label+'.stderr.log')).open('x') as err:subprocess.run(c,check=True,stdout=out,stderr=err,timeout=timeout)
 print('PASS',label,flush=True)
head=subprocess.check_output(['git','ls-remote','--exit-code',remote,'refs/heads/master'],text=True).split()[0];assert len(head)==40
version=base/'versions'/('master-'+head);assert not version.exists(),'existing version must be inspected, never overwritten';source=version/'source';version.mkdir()
run('central-clone',['git','clone','--depth','1','--branch','master',remote,str(source)],300)
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==head
build=Path(tempfile.mkdtemp(prefix='gem16-central-llama-build-',dir='/tmp'));manifest={'commit':head,'source':str(source),'build_directory':str(build),'reason':'build intermediates in tmpfs; permanent source/runtime retained centrally','commands':{}}
for profile in ['cpu','cuda']:
 b=build/profile;opts=['-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=ON','-DLLAMA_BUILD_TOOLS=ON','-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON']
 opts+=['-DGGML_CUDA=OFF','-DGGML_NATIVE=ON'] if profile=='cpu' else ['-DGGML_CUDA=ON','-DGGML_NATIVE=OFF','-DCMAKE_CUDA_ARCHITECTURES=120a-real','-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc']
 run('central-configure-'+profile,['cmake','-S',str(source),'-B',str(b),*opts],300)
 run('central-build-'+profile,['cmake','--build',str(b),'--parallel','6','--target','llama-cli','llama-bench','llama-quantize','llama-server','llama-gguf'],1800)
 dest=source/('build-central-'+profile)/'bin';dest.parent.mkdir();shutil.copytree(b/'bin',dest,symlinks=True)
 text=subprocess.check_output([str(dest/'llama-server'),'--version'],stderr=subprocess.STDOUT,text=True);assert head[:7] in text;manifest[profile]={'version':text,'binaries':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file() and not p.is_symlink()}}
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
run('central-device-check',[str(source/'build-central-cuda/bin/llama-cli'),'--list-devices'],60);assert 'CUDA0' in (O/'central-device-check.stdout.log').read_text()+(O/'central-device-check.stderr.log').read_text()
manifest['built_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(version/'manifest.txt').write_text(json.dumps(manifest,indent=2)+'\n');(version/'.complete').touch()
latest=subprocess.check_output(['git','ls-remote','--exit-code',remote,'refs/heads/master'],text=True).split()[0];assert latest==head,'upstream advanced during build; completed immutable version retained, current unchanged'
link=base/('.current-ordinary-'+str(os.getpid()));os.symlink('versions/master-'+head+'/source',link);os.replace(link,base/'current')
(O/'central-build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('CENTRAL CURRENT',head,flush=True)
