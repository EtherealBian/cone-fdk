"""Build all four native CLIs on DGX Spark; preserve frozen Windows source/product receipts."""
import json, os, platform, shutil, subprocess, sys, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,help='Build in a new directory; do not change checkout aliases')
options=parser.parse_args(); OUT=options.out.resolve() if options.out else ROOT/'build/spark'
if options.out: OUT.mkdir(parents=True,exist_ok=False)
SRC=OUT/'src'; SRC.mkdir(parents=True,exist_ok=True)
def one(text,old,new):
    if text.count(old)!=1: raise ValueError(f'unrecognized backend boundary: {old!r}')
    return text.replace(old,new)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def call(args): subprocess.run(list(map(str,args)),check=True)
if sys.platform!='linux' or platform.machine()!='aarch64': raise SystemExit('DGX Spark build requires Linux aarch64')
for path in (ROOT/'src').iterdir():
    dest=SRC/path.name
    if path.name in ('files.hpp','finite.hpp'):
        shutil.copyfile(HERE/({'files.hpp':'files_posix.hpp','finite.hpp':'finite_portable.hpp'}[path.name]),dest);continue
    if path.name=='gpu.hpp':
        s=one(path.read_text(),'#include <windows.h>','#include <dlfcn.h>');a=s.index('struct Lib {');b=s.index('struct Gpu {',a);s=s[:a]+(HERE/'lib_linux.hpp').read_text()+s[b:];dest.write_text(one(one(s,'"nvcuda.dll"','"libcuda.so.1"'),'"cufft64_11.dll"','"libcufft.so.12"'))
    elif path.suffix=='.cpp':
        s=one(one(path.read_text(),'int main()','int main(int nargs, char** values)'),'arguments()','std::vector<std::string>(values,values+nargs)');
        if path.stem.startswith('periodic_'):s=one(s,'GetFileAttributesW(dst.c_str()) != INVALID_FILE_ATTRIBUTES','path_exists(dst)')
        if path.stem.startswith('periodic_'):
            var='out' if path.stem=='periodic_labels' else 'output'
            s=one(s,f'MoveFileExW({var}.tmp.c_str(), dst.c_str(), MOVEFILE_WRITE_THROUGH)',f'publish_fresh({var}.tmp, dst)')
        dest.write_text(s)
    else: shutil.copyfile(path,dest)
nvcc=os.environ.get('NVCC','/usr/local/cuda/bin/nvcc'); cxx=os.environ.get('CXX','g++'); products=[]
for name,source in [('cone-fdk','main'),('wall-pack','wall_pack'),('periodic-labels','periodic_labels'),('periodic-interfaces','periodic_interfaces')]:
    target=OUT/name; call([cxx,'-O3','-std=c++14',SRC/(source+'.cpp'),'-ldl','-o',target]);products.append(target)
for name in ('kernels','periodic_labels','periodic_interfaces'):
    t=OUT/(name.replace('_','-')+'.ptx'); call([nvcc,'-ptx','-arch=compute_121','--std=c++17',SRC/(name+'.cu'),'-o',t]);products.append(t)
receipt={'machine':platform.machine(),'nvcc':subprocess.check_output([nvcc,'--version'],text=True).strip(),'compiler':subprocess.check_output([cxx,'--version'],text=True).splitlines()[0],'arch':'compute_121','products':{p.name:digest(p) for p in products}}
(OUT/'build.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt,indent=2))

if not options.out:
    # Historical callers use .exe in the name, never a Windows executable on Spark.
    for target in products:
        alias=OUT.parent/(target.name if target.suffix=='.ptx' else target.name+'.exe')
        wanted=Path('spark')/target.name
        if alias.is_symlink() and alias.readlink()==wanted: continue
        alias.symlink_to(wanted)  # fail closed if a different path already exists
