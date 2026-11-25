"""Qualify a GB10 publisher overlay with archived, already-certified simulated maps."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OVERLAY = ROOT/'build/spark/python'
from build_publishers import build as check_overlay


def run(archive, build, qualification, work):
    overlay_pin = check_overlay(OVERLAY)
    sys.path[:0] = [str(OVERLAY), str(ROOT/'scripts')]
    import periodic_interfaces as interfaces
    import periodic_parcel as parcel
    import periodic_roi as roi
    from anchor_audit import publish
    from build_periodic_labels import sha
    from qualify_interface_parcels import qualify as faces
    from qualify_periodic_labels import checked, need
    from qualify_periodic_roi import CAMPAIGN, qualify as crops

    for module in (interfaces, parcel, roi):
        need(Path(module.__file__).resolve().parent == OVERLAY, 'publisher module shadowed')
    reps = ROOT/'reports'; survey = reps/'periodic-parcels.json'
    checked(survey,CAMPAIGN)
    expected_faces = checked(reps/'interface-parcels.json', '227b105a5890f3964a7b72bc439940ca39aa9ee607cc9355122928807c9e3ac6')
    expected_crops = checked(reps/'periodic-roi-qualification.json', '8d1fcde0c68924db3de6a9158c37d10a6a7ca180a793e9ce89630fff4d34da63')
    reference = json.loads((reps/'spark-20261125/interface-comparison.json').read_bytes())
    device = checked(qualification, reference['spark_qualification_sha256'])
    need(device['gpu']=='NVIDIA GB10', 'wrong device qualification')
    work.mkdir(mode=0o700,exist_ok=False)
    faces(survey, archive, build, qualification, work/'faces')
    crops(survey, archive, work/'roi')
    fs = json.loads((work/'faces/qualification.json').read_bytes())
    cs = json.loads((work/'roi/qualification.json').read_bytes())
    for k in ('maps','paired_cells','faces_by_direction_and_kind','parcel_campaign_sha256'):
        need(fs[k]==expected_faces[k], 'different archived interface '+k)
    need(set(fs['cases'])==set(expected_faces['cases']), 'different cases')
    for name,actual in fs['cases'].items():
        exp = expected_faces['cases'][name]
        need(actual['status']==exp['status'] and len(actual['rows'])==len(exp['rows']), 'case coverage')
        for new,old in zip(actual['rows'], exp['rows'],strict=True):
            # Spark's outer publisher receipt has new device and build identities by design.
            need({k:v for k,v in new.items() if k!='result_manifest_sha256'} ==
                 {k:v for k,v in old.items() if k!='result_manifest_sha256'}, 'changed faces or identities')
    for k in ('parcels','partition_cells','partition_and_point_queries','upstream_refused','cases'):
        need(cs[k]==expected_crops[k], 'different authenticated crop '+k)
    for k in ('blind','seam'):
        need(cs['portable_examples'][k]['receipt']==expected_crops['portable_examples'][k]['receipt'], 'detached receipt differs')
    publish(work/'comparison.json', dict(format='CFDK-Spark-periodic-campaign-v1',
        hardware=device['gpu'], overlay_sha256=overlay_pin, generator_sha256=sha(Path(__file__)),
        # Source lists inside upstream reports identify frozen inputs; this sidecar binds the production modules actually loaded.
        runtime_modules={m.__name__:sha(Path(m.__file__)) for m in (interfaces,parcel,roi)},
        archived_survey_sha256=CAMPAIGN, device_qualification_sha256=sha(qualification),
        reference_reports={n:sha(reps/n) for n in ['interface-parcels.json','periodic-roi-qualification.json']},
        reports={k:sha(work/k/'qualification.json') for k in ['faces','roi']},
        maps=fs['maps'], cells=fs['paired_cells'], crop_queries=cs['partition_and_point_queries'], refused=cs['upstream_refused'],
        detached_examples=sorted(cs['portable_examples']), faces_by_direction_and_kind=fs['faces_by_direction_and_kind'],
        scope='Archived simulated threshold-pair maps; every face and canonical component checked, not independent scans or physical defects.'))
    print((work/'comparison.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('archive','build','qualification','work'): p.add_argument(n,type=Path)
    run(**{k:v.absolute() for k,v in vars(p.parse_args()).items()})
