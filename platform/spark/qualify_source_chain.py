"""Rebuild all accepted label parcels from wall records on Spark, then use their NEW roots."""
import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OVERLAY=ROOT/'build/spark/python'
from build_publishers import build as check_overlay


def run(recipe, archive, labeljob, ifacejob, work):
    overlay=check_overlay(OVERLAY);sys.path[:0]=[str(OVERLAY),str(ROOT/'scripts')]
    import numpy as np
    import periodic_interfaces as iface
    import periodic_parcel as pp
    import periodic_roi as roi
    from anchor_audit import publish
    from build_periodic_labels import sha
    from qualify_interface_parcels import oracle
    from qualify_periodic_labels import checked,need
    from qualify_periodic_parcels import qualify
    from qualify_periodic_roi import CAMPAIGN,check as direct

    for mod in [iface,pp,roi]:need(Path(mod.__file__).parent==OVERLAY,'wrong runtime modules')
    reports=ROOT/'reports';survey=checked(reports/'periodic-parcels.json',CAMPAIGN)
    checked(archive/'result.json',CAMPAIGN);of=checked(reports/'interface-parcels.json','227b105a5890f3964a7b72bc439940ca39aa9ee607cc9355122928807c9e3ac6')
    lq=labeljob/'cases/qualification.json';fq=ifacejob/'cases/qualification.json'
    lpin=json.loads((reports/'spark-20261125/label-comparison.json').read_bytes())['qualification_sha256']
    fpin=json.loads((reports/'spark-20261125/interface-comparison.json').read_bytes())['spark_qualification_sha256']
    for path,pin in [(lq,lpin),(fq,fpin)]:need(checked(path,pin)['gpu']=='NVIDIA GB10','wrong qualified GPU')
    work.mkdir(mode=0o700,exist_ok=False);parcels=work/'parcels'
    qualify(recipe,reports,labeljob/'build',lq,parcels)
    new=json.loads((parcels/'result.json').read_bytes());fs=work/'faces';fs.mkdir();indexes=work/'roi';indexes.mkdir()
    for k in ('recipe_sha256','packed_crosswalk','references','maps','pair_cells'):
        need(new[k]==survey[k], 'different upstream '+k)
    need(set(new['cases'])==set(survey['cases']), 'different cases')
    rows=[];refused=[]
    for name,case in survey['cases'].items():
        after=new['cases'][name]; need(after['status']==case['status'], 'status changed')
        if case['status']=='upstream_refused':
            need(case==after and not (parcels/name).exists() and not (fs/name).exists(), 'incorrect refusal');refused.append(name);continue
        (fs/name).mkdir();(indexes/name).mkdir()
        for old,row,expected_face in zip(case['rows'],after['rows'],of['cases'][name]['rows'],strict=True):
            a,b=row['pair'];n=f'{a}-{b}'; op=old['manifest_sha256'];npin=row['manifest_sha256']; oldroot=archive/name/n;nr=parcels/name/n
            need(npin!=op and old['pair']==row['pair']==expected_face['pair'],'missing new provenance')
            olddoc=checked(oldroot/'manifest.json',op);newdoc=checked(nr/'manifest.json',npin)
            meas=lambda obj:{k:v for k,v in obj.items() if k not in ('qualification_sha256','backend')}
            need(meas(olddoc)==meas(newdoc) and newdoc['backend']['gpu']=='NVIDIA GB10' and newdoc['qualification_sha256']==lpin,'different certificate/source scope')
            need({k:v for k,v in old.items() if k!='manifest_sha256'}=={k:v for k,v in row.items() if k!='manifest_sha256'},'different masks, labels or anchor state')
            # Negative provenance: old pin must not be accepted even for identical measurement bytes.
            try:pp.offline(nr,op,7)
            except ValueError:pass
            else:raise ValueError('old parent unexpectedly accepted')
            fd=fs/name/n; face=iface.generate(nr,fd,ifacejob/'build',fq,npin,fpin,7,1)
            rp=sha(fd/'manifest.json');result=iface.verify(nr,fd,npin,rp,1);s=result['summary'];h,w=result['shape_ha']
            mask=np.fromfile(nr/'mask.u8','u1').reshape(h,w);lab=np.fromfile(nr/'labels.i32','<i4').reshape(h,w)
            need(face['measurement']==result and s['components']==oracle(mask,lab) and s['face_u8_sha256']==expected_face['face_sha256'] and s['faces_by_direction_and_kind']==expected_face['faces'], 'independent face comparison')
            idx=indexes/name/(n+'.json');ip=roi.index(nr,idx,npin,4)
            boxes=[[0,w-1,h//2,2],[h-1,0,1,w]]
            for box in boxes:direct(roi.inspect(nr,idx,npin,ip,box),mask,lab,box)
            rows.append(dict(case=name,pair=row['pair'],old_manifest_sha256=op,new_manifest_sha256=npin,mask_sha256=row['mask_sha256'],labels_sha256=row['labels_sha256'],face_manifest_sha256=rp,face_sha256=s['face_u8_sha256'],index_sha256=ip,checked_crops=boxes))
    need(len(rows)==67 and len(refused)==2,'incomplete chain')
    publish(work/'chain.json',dict(format='CFDK-GB10-source-to-interface-v1',platform='NVIDIA GB10 / Linux aarch64',
        source_recipe_sha256=sha(recipe),archived_survey_sha256=CAMPAIGN,spark_survey_sha256=sha(parcels/'result.json'),
        overlay_sha256=overlay,harness_sha256=sha(Path(__file__)),label_qualification_sha256=lpin,interface_qualification_sha256=fpin,
        maps=len(rows),cells=new['pair_cells'],new_lineage_crop_queries=2*len(rows),old_root_refusals=len(rows),upstream_refusals=refused,rows=rows,
        scope='Eight simulated projection-derived comparison jobs plus two withheld jobs; not eight measured specimens or physical accuracy.'))
    print('Complete independent Spark source chain:',len(rows),'maps,',len(refused),'refusals',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('recipe','archive','labeljob','ifacejob','work'):p.add_argument(n,type=Path)
    run(**{k:v.absolute() for k,v in vars(p.parse_args()).items()})
