"""Re-run every eligible archived witness shell pair on GB10 before rebuilding all downstream evidence.

Only novel lineage manifests survive: generated payloads must match the raw archive store
before temporary files close. Nominal walls are authenticated parent data, not rerun here.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import witness_lineage as l
from build_publishers import build
from spark_publication import fresh, publish, staging

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OVER = ROOT / "build/spark/python"
ATLAS = "99debd530f6c431f5ae4a7a2bda91390f9a94392a01a49edc59a4ea541bfe601"
OVERPIN = "0434f703e71e4218dd9ca403a5a73de423c2b8e07b4b47e0036b608ddb23faaf"


def run(dest, store, labels, interfaces):
    import numpy as np

    l.need(build(OVER) == OVERPIN, "unqualified publisher overlay")
    sys.path[:0] = [str(OVER), str(ROOT / "scripts")]
    import periodic_parcel as pp
    import periodic_interfaces as face
    import periodic_roi as roi
    import witnessed_wall as w
    import threshold_atlas as atlas
    from qualify_threshold_atlas import periodic_ids
    from qualify_interface_parcels import oracle
    from qualify_periodic_roi import check as crop_check
    from validate_edges import raw_oracle

    l.need(
        all(Path(mod.__file__).parent == OVER for mod in (roi, pp, face))
        and w.sha(HERE / "witnessed_wall.py") == l.ADAPTER
        and w.sha(ROOT / "scripts/validate_edges.py")
        == "d749599db9d48d7ac1029b1f3b8b1dda0c868d4449ddad0766d8e3e69df3c3c7",
        "unqualified code",
    )
    dest = fresh(dest)
    store = store.resolve(strict=True)
    l.need(not dest.is_relative_to(store), "write outside archive")
    q = labels / "cases/qualification.json"
    iq = interfaces / "cases/qualification.json"
    l.read(q, l.LABEL_Q, 4 * l.MIB)
    l.read(iq, l.FACE_Q, 4 * l.MIB)
    survey = json.loads(l.obj(store, l.CAMPAIGN))
    faces = json.loads(l.obj(store, l.FACES))
    plan = json.loads(l.obj(store, survey["recipe_sha256"]))
    prior = json.loads(
        l.read(ROOT / "reports/spark-20261125/threshold-atlas/atlas.json", ATLAS, 4 * l.MIB)
    )
    cases = {}
    with staging(dest, ".witness-lineage-") as tmp:
        private = Path(tmp)
        out = private / "delta"
        out.mkdir()

        def save(p):
            data = p.read_bytes()
            h = l.sha(data)
            t = out / h
            if t.exists():
                l.read(t.resolve(strict=True), h, 4 * l.MIB)
            else:
                with t.open("xb") as f:
                    f.write(data)
            return h

        for e in plan["entries"]:
            name = e["id"]
            ep = e["manifest_sha256"]
            old = json.loads(l.obj(store, ep))
            base = survey["cases"][name]
            with TemporaryDirectory(dir=private.resolve(strict=True)) as t:
                tmpcase = Path(t)
                edge = tmpcase / "review"
                if e["kind"] == "witness-refusal":
                    try:
                        w.produce(store, ep, edge, [0.018, 0.02, 0.022], 1)
                    except ValueError as err:
                        l.need(
                            str(err) == "parent is not an accepted witness measurement"
                            and not edge.exists(),
                            "incorrect witness refusal",
                        )
                    else:
                        raise ValueError("withheld data published")
                    l.need(
                        old["reason"] == base["reason"] and base["status"] == "upstream_refused",
                        "refusal changed",
                    )
                    cases[name] = dict(status="withheld", reason=old["reason"], parent_sha256=ep)
                    continue
                wp = old["inputs_sha256"]["manifest.json"]
                new = w.produce(store, wp, edge, old["levels_per_mm"], 1)
                npin = save(edge / "manifest.json")
                l.need(l.wall(old, new, ep, npin) == wp, "upstream crosswalk")
                par = json.loads(l.obj(store, wp))
                view = tmpcase / "raw"
                view.mkdir()
                for n in ["material.grid", "inspected/shells.f32", "reference/shells.f32"]:
                    h = par["outputs_sha256"][n]
                    l.obj(store, h, 256 * l.MIB)
                    f = view / n
                    f.parent.mkdir(exist_ok=True)
                    f.symlink_to(w.object_path(store, h))
                measurement = raw_oracle(view, edge, new)
                for n, h in new["outputs_sha256"].items():
                    l.need(
                        (edge / n).read_bytes() == l.obj(store, h, 32 * l.MIB),
                        "archive wall bytes differ",
                    )
                frames = {}
                rows = []
                shape = new["shape_ha"]
                hh, ww = shape
                for p, fr in zip(base["rows"], faces["cases"][name]["rows"], strict=True):
                    pair = p["pair"]
                    l.need(pair == fr["pair"], "historical pair mismatch")
                    a = json.loads(l.obj(store, p["manifest_sha256"]))
                    fd = json.loads(l.obj(store, fr["result_manifest_sha256"]))
                    anchors = e["anchors_row_col"]
                    with TemporaryDirectory(dir=tmpcase) as td:
                        path = Path(td)
                        parcel = path / "parcel"
                        fp = path / "faces"
                        idx = path / "index"
                        b = pp.generate(
                            edge,
                            parcel,
                            npin,
                            labels / "build",
                            q,
                            l.LABEL_Q,
                            pair,
                            anchors,
                            e["band_rows"],
                            32,
                        )
                        bp = save(parcel / "manifest.json")
                        l.parcel(a, b, p["manifest_sha256"], bp, ep, npin, pair)
                        pp.offline(parcel, bp, 3)
                        f = face.generate(
                            parcel, fp, interfaces / "build", iq, bp, l.FACE_Q, 7, 1
                        )
                        fh = save(fp / "manifest.json")
                        l.face(
                            fd,
                            f,
                            fr["result_manifest_sha256"],
                            fh,
                            ep,
                            npin,
                            p["manifest_sha256"],
                            bp,
                        )
                        s = face.verify(parcel, fp, bp, fh, 1)["summary"]
                        mb = (parcel / "mask.u8").read_bytes()
                        lb = (parcel / "labels.i32").read_bytes()
                        fb = (fp / "face.u8").read_bytes()
                        l.need(
                            all(
                                b0 == l.obj(store, h)
                                for b0, h in (
                                    (mb, p["mask_sha256"]),
                                    (lb, p["labels_sha256"]),
                                    (fb, fr["face_sha256"]),
                                )
                            ),
                            "historical payload differs",
                        )
                        m = np.frombuffer(mb, "u1").reshape(shape)
                        labs = np.frombuffer(lb, "<i4").reshape(shape)
                        np.testing.assert_array_equal(labs, periodic_ids(m))
                        l.need(s["components"] == oracle(m, labs), "independent face tally")
                        ix = roi.index(parcel, idx, bp, 4)
                        l.need(save(idx) == ix, "index bytes")
                        l.index_blocks(json.loads(idx.read_bytes()), bp, shape, mb, lb)
                        for box in ([0, ww - 2, hh, 4], [hh - 1, 0, 1, ww]):
                            crop_check(roi.inspect(parcel, idx, bp, ix, box), m, labs, box)
                        frames[tuple(pair)] = (mb, labs.ravel().tolist())
                        rows.append(dict(pair=pair, parcel=bp, face=fh, index=ix))
                actual, _ = atlas.chart(frames, hh, ww, len(new["levels_per_mm"]), anchors, [])
                ref = prior["cases"][name]
                l.need(
                    l.equal(
                        actual,
                        {
                            k: v
                            for k, v in ref.items()
                            if k not in ("source", "parcel_sha256_by_pair")
                        },
                    ),
                    "threshold atlas changed",
                )
                cases[name] = dict(
                    status="accepted",
                    old_edge=ep,
                    edge=npin,
                    parent=wp,
                    raw_oracle=measurement,
                    atlas_numeric_sha256=l.sha(l.encode(actual)),
                    rows=rows,
                )
                print(name, "ok", len(rows), flush=True)
        d = dict(
            format="CFDK-GB10-witness-lineage-v1",
            scope="Frozen simulated radial-shell data and nominal walls; 8 comparison jobs, not measured defects. Authenticated historical payloads are necessary for standalone verification.",
            frozen_campaign=l.CAMPAIGN,
            frozen_interfaces=l.FACES,
            frozen_atlas=ATLAS,
            qualifier_sha256=w.sha(Path(__file__)),
            offline_reviewer_sha256=w.sha(HERE / "witness_lineage.py"),
            cases=cases,
        )
        (out / "campaign.json").write_bytes(l.encode(d) + b"\n")
        root = w.sha(out / "campaign.json")
        physical = out.resolve(strict=True)
        summary = l.audit(physical, store, root)
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(HERE / "witness_lineage.py"),
                str(physical),
                str(store),
                root,
            ],
            capture_output=True,
            check=True,
            cwd=physical,
        )
        l.need(json.loads(p.stdout) == summary and not p.stderr, "isolated closure")
        publish(out, dest)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    [p.add_argument(n, type=Path) for n in ("dest", "store", "labels", "interfaces")]
    run(**{k: v.absolute() for k, v in vars(p.parse_args()).items()})
