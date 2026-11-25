"""Pine measured-scan qualification. Never substitute a phantom or one archive for a source."""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
from prepare_pine import HERE, ROOT, pine
from sealed_inputs import publish_new, snapshot
from common import Geometry, astra_run, difference, reconstruct

need, sha = pine.need, pine.sha
BUILD_PIN = '41e1ba90f3ca5422b50f3d34f711ac4112aa14fa4d4fab5fed673fbe03b33b16'


def checked(path,pin,limit):
    with snapshot(path,pin,limit=limit) as fd:
        with os.fdopen(os.dup(fd),'rb') as f: return json.load(f)


def provenance(folder, scan_pin, original_receipt):
    protocol=checked(ROOT/'fixtures/pine-cone-qualification-protocol-v1.json',pine.RULES,1<<16)
    need(protocol['source_manifest_sha256']==pine.SOURCE,'different source protocol')
    src=pine.source(); d=checked(folder/'scan.json',scan_pin,1<<16)
    need(d['format']=='CFDK-pine-cone-preprocessing-v1' and d['protocol_sha256']==pine.RULES and
         d['source_manifest_sha256']==pine.SOURCE and d['used_original_frames']==720 and
         d['duplicate_360_excluded'] is True and d['independently_rechecked_sparse_views']==90,'incomplete preprocessing')
    need(d['script_sha256']==sha(ROOT/'scripts/prepare_pine_cone.py') and
         d['platform_sources']=={n:sha(HERE/n) for n in ('prepare_pine.py','sealed_inputs.py')},'preprocessor changed')
    digests=d['original_file_sha256']; need(set(digests)==set(src['files']) and
          all(re.fullmatch('[0-9a-f]{64}',v) for v in digests.values()) and
          digests[pine.META]==src['acquisition_metadata_sha256'],'invalid originals')
    need(checked(original_receipt,d['source_receipt_sha256'],1<<16)==dict(source_manifest_sha256=pine.SOURCE,full_file_sha256=digests),'original acquisition receipt differs')
    frames=checked(folder/'frames.json',d['frame_manifest_sha256'],1<<20)
    need(len(frames)==721,'frame count')
    for k in range(1,722):
        frame=frames[f'20201118_pine_cone_{k:04d}.tif'];need(frame['used'] is (k!=721) and frame['archive'] in digests and frame['archive']!=pine.META and re.fullmatch('[0-9a-f]{64}',frame['sha256']) and re.fullmatch('[0-9a-f]{8}',frame['crc32']),'invalid frame provenance')
    need(set(d['scans'])=={'pine90','pine720'} and d['scans']['pine90']['shape_vu']==d['scans']['pine720']['shape_vu'],'different scan ordering or shape')
    return protocol,d


@contextmanager
def inputs(folder, build, row, name, binding, products):
    views=int(name.removeprefix('pine'));h,w=row['shape_vu']; nbytes=4*views*h*w
    need(views in (90,720) and row['views']==views and type(h) is int and type(w) is int and
         16<=h<=512 and 16<=w<=512 and h*w<=131072 and row['projection_bytes']==nbytes,'invalid geometry or projection length')
    with ExitStack() as stack:
        for original, alias, pin, limit, exact, execute in (
          (folder/(name+'.geom'),'scan.geom',row['geom_sha256'],1<<16,None,False),
          (folder/(name+'.f32'),'scan.f32',row['projections_sha256'],512<<20,nbytes,False),
          (build/'cone-fdk','cone-fdk.exe',products['cone-fdk'],64<<20,None,True),
          (build/'kernels.ptx','kernels.ptx',products['kernels.ptx'],64<<20,None,False)):
            fd=stack.enter_context(snapshot(original,pin,limit=limit,exact=exact,executable=execute));(binding/alias).symlink_to(f'/proc/{os.getpid()}/fd/{fd}')
        g=Geometry.read(binding/'scan.geom')
        need(g==Geometry(views,h,w,.4,.4,230.66,553.74),'not the untuned pine acquisition geometry')
        yield binding/'scan'


def guards(d,n,views,fov,budget,batch,box,projection_bytes):
    wanted=4*math.prod(box[3:]);need(d['gpu']=='NVIDIA GB10' and d['views']==views and d['n']==n and
        d['batch']==d['requested_batch']==batch and d['budget_bytes']==budget<<20 and d['redzones_verified'] is True and
        0<d['peak_owned_bytes']<=d['effective_budget_bytes']<=d['budget_bytes'] and d['roi_origin_xyz']==box[:3] and
        d['output_shape_xyz']==box[3:] and d['output_bytes']==wanted and d['voxel_spacing_mm']==fov/n and
        d['projection_passes']>=1 and d['projection_bytes_read']==projection_bytes*d['projection_passes'],'invalid device accounting or ROI')


def volume(p,shape):
    need(p.is_file() and p.stat().st_size==4*math.prod(shape),'wrong volume size')
    v=np.memmap(p,'<f4','r',shape=shape)
    need(np.isfinite(v).all() and float(v.max())>float(v.min()),'nonfinite or constant reconstruction')
    return v


def check_roi(full, output, box):
    x,y,z,w,h,d=box; need(output.stat().st_size==4*d*h*w,'ROI length')
    true=full[z:z+d,y:y+h,x:x+w].tobytes(); need(output.read_bytes()==true,'ROI differs from dense crop')
    return hashlib.sha256(true).hexdigest()


def measure(scan,binding,dest,views,row,settings):
    n=settings['n'];fov=settings['fov_mm'];budget=settings['budget_mib'];batch=settings['batch'];boxes=settings['rois_xyzwhd']
    dest.mkdir();fullbox=[0,0,0,n,n,n]; runs=[]
    for k in (0,1):
        file=dest/f'native-{k}.f32'; run=reconstruct(scan,file,n,fov,budget,batch,build=binding)
        guards(run,n,views,fov,budget,batch,fullbox,row['projection_bytes']); volume(file,(n,n,n)); runs.append(run)
    first,repeat=dest/'native-0.f32',dest/'native-1.f32'; native_sha=sha(first)
    need(native_sha==sha(repeat),'native runs differ');full=volume(first,(n,n,n)); roi=[]
    for k,box in enumerate(boxes):
        out=dest/f'roi-{k}.f32';run=reconstruct(scan,out,n,fov,budget,batch,build=binding,roi=','.join(map(str,box)))
        guards(run,n,views,fov,budget,batch,box,row['projection_bytes']);roi.append(dict(box=box,sha256=check_roi(full,out,box),run=run))
    ref=dest/'astra.f32'; ar=astra_run(scan,ref,n,fov); volume(ref,(n,n,n));diff=difference(first,ref,n)
    need(all(math.isfinite(x) for x in diff.values()) and diff['rel_l2']<.002,'unchanged-scale independent ASTRA comparison failed')
    # The second dense result was freshly rehashed; keep its receipt and the identical first copy.
    need(sha(repeat)==native_sha,'repeat changed before cleanup'); repeat.unlink()
    return dict(views=views,projections_sha256=row['projections_sha256'],geometry_sha256=row['geom_sha256'],native_runs=runs,
                dense_sha256=native_sha,repeat_sha256=native_sha,voxels=n**3,astra=ar,astra_sha256=sha(ref),native_vs_astra=diff,
                rois=roi,retention={'native-1.f32':'fresh identical repeat; retained native-0.f32'})


def run(scans,original_receipt,build,work,scan_pin):
    from replay_spark import private_location
    import astra
    work=private_location(work);need(not os.path.lexists(work),'existing output')
    need(all(not work.is_relative_to(p.resolve()) and not p.resolve().is_relative_to(work) for p in (scans,build,original_receipt)), 'output intersects an input')
    protocol,doc=provenance(scans,scan_pin,original_receipt); rec=checked(build/'build.json',BUILD_PIN,1<<16)
    need('NVIDIA GB10' in astra.get_gpu_info(0),'wrong ASTRA device')
    with tempfile.TemporaryDirectory(dir=work.parent,prefix='.pine-qual-') as t:
        temp=Path(t);stage=temp/'output';stage.mkdir();rows={}
        for name in ('pine90','pine720'):
            bound=temp/name;bound.mkdir();row=doc['scans'][name]
            with inputs(scans,build,row,name,bound,rec['products']) as scan:
                rows[name]=measure(scan,bound,stage/name,row['views'],row,protocol['reconstruction'])
        pine.publish(stage/'qualification.json',dict(format='CFDK-Spark-physical-pine-qualification-v1',
            source_manifest_sha256=pine.SOURCE,protocol_sha256=pine.RULES,scan_receipt_sha256=scan_pin,
            frame_manifest_sha256=doc['frame_manifest_sha256'],build_sha256=BUILD_PIN,script_sha256=sha(Path(__file__)),
            independent_code_sha256=sha(ROOT/'scripts/common.py'),gpu=astra.get_gpu_info(0),scans=rows,
            scope=protocol['no_physical_claim']))
        publish_new(stage,work)
    print('Verified both measured scans and four direct ROIs:',sha(work/'qualification.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('scans','original_receipt','build','work'):p.add_argument(n,type=Path)
    p.add_argument('--scan-pin',required=True);run(**vars(p.parse_args()))
