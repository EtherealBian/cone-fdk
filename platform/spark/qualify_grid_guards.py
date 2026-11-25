"""Mixed parent grids on adversarial signed fixtures; check refusal, pack placement and CUDA sanitizers."""

import argparse, hashlib, json, os, struct, subprocess, tempfile
from pathlib import Path
import qualify_tiles as gen
import spark_multigrid as s
from spark_multigrid import M, g, q, sha, need, fresh, staging, publish, v

GRIDS = [
    dict(n=63, fov_mm=18.4, box_xyz=[0, 0, 0, 63, 63, 63]),
    dict(n=65, fov_mm=14.7, box_xyz=[1, 2, 3, 29, 31, 23]),
    dict(n=127, fov_mm=24.3, box_xyz=[1, 2, 3, 29, 31, 23]),
    dict(n=41, fov_mm=17.9, box_xyz=[40, 38, 37, 1, 3, 4]),
]


def run(dest):
    dest = fresh(dest)
    _ = s.certificate()
    cases = []
    refusals = []
    tools = []
    with staging(dest, ".multigrid-guards-") as td:
        stage = Path(td) / "ready"
        stage.mkdir()
        wd = Path(td).resolve() / "scratch"
        wd.mkdir(mode=0o700)
        geo, pr = gen.inputs(wd)
        gridfile = wd / "grids.txt"
        origgeo = geo.read_bytes()
        gp = s.BDIR / "multigrid"
        ptx = g.ROOT / "build/spark/kernels.ptx"
        need(
            sha(gp) == s.certificate()[1]["builds"][0]["multigrid"] and sha(ptx) == q.PTX,
            "binary mismatch",
        )
        known = None
        for label, reverse, positive in [
            ("reverse-odd", False, False),
            ("reverse-reordered", True, False),
            ("forward-origin", False, True),
        ]:
            if positive:
                geo.write_bytes(b"CFDK1 13 33 37 0.7 0.8 100 150 -0.631 1\n")
            grids = list(reversed(GRIDS)) if reverse else GRIDS
            req = dict(
                format="CFDK-multigrid-request-v1",
                geometry_sha256=sha(geo),
                projections_sha256=sha(pr),
                batch=8,
                cap_mib=4,
                grids=grids,
            )
            path = wd / f"{label}.json"
            path.write_text(json.dumps(req) + "\n")
            dst = wd / label
            d = s.issue(dst, user=(path, sha(path), geo, pr))
            keys = [row["sha256"] for row in d["pack"]["regions"]]
            need(
                d["mode"] == "fresh-original-per-grid"
                and keys == [r["sha256"] for r in d["independent"]]
                and len(set(keys)) == 4,
                "redundant or unchecked fixture",
            )
            pack = dst / "volume.mg"
            need(v.verify(pack, d["pack"]["pack_sha256"]) == d["pack"], "cannot reopen")
            if known is None:
                known = keys
            elif reverse:
                need(known == keys[::-1], "grid identity tied to iteration order")
            else:
                need(keys != known, "different acquisition ignored")
            nrrd = wd / f"{label}.nrrd"
            export = 1
            row = d["pack"]["regions"][export]
            v.verify(pack, d["pack"]["pack_sha256"], export, nrrd)
            header, payload = nrrd.read_bytes().split(b"\n\n", 1)
            need(
                header + b"\n\n" == v.nrrd(row)
                and len(payload) == row["size_bytes"]
                and hashlib.sha256(payload).hexdigest() == row["sha256"],
                "not a correct NRRD export",
            )
            cases.append(
                dict(
                    label=label,
                    request=req,
                    request_sha256=sha(path),
                    nrrd_sha256=sha(nrrd),
                    receipt=d,
                )
            )
        geo.write_bytes(origgeo)
        gr = s.validate_recipe(dict(cases[0]["request"]))
        gridfile.write_bytes(q.spec(gr))
        raw = wd / "trial.mg"
        base = [
            str(gp),
            "--geom",
            str(geo),
            "--input",
            str(pr),
            "--ptx",
            str(ptx),
            "--grids",
            str(gridfile),
            "--output",
            str(raw),
            "--budget",
            "4",
            "--batch",
            "8",
        ]
        expected = cases[0]["receipt"]["pack"]["pack_sha256"]
        for tool in ["memcheck", "synccheck"]:
            log = stage.resolve() / f"{tool}.log"
            p = subprocess.run(
                [
                    "/usr/local/cuda/bin/compute-sanitizer",
                    "--tool",
                    tool,
                    "--error-exitcode",
                    "79",
                    "--log-file",
                    str(log),
                    *base,
                ],
                capture_output=True,
                timeout=180,
            )
            text = log.read_text() if log.exists() else ""
            need(
                not p.returncode and not p.stderr and "ERROR SUMMARY: 0 errors" in text,
                "sanitizer failure " + text[-300:] + p.stderr.decode(errors="replace")[-200:],
            )
            native = json.loads(p.stdout)
            q.guard(native, (13, pr.stat().st_size), 4, 8, gr, True)
            need(sha(raw) == expected, "instrumented output differs")
            raw.unlink()
            tools.append(dict(tool=tool, log_sha256=sha(log), pack_sha256=expected))
        for fault, msg in [
            ("cap", "one pass"),
            ("nan-tail", "non-finite line integral"),
            ("exists", "cannot publish fresh"),
            ("alias", "aliases grid list"),
            ("extra", "unknown/duplicate"),
            ("bad-box", "ROI outside"),
            ("empty", "empty grid list"),
            ("nonfinite-pitch", "expect n fov_mm"),
            ("seventeenth", "at most 16"),
        ]:
            args = list(base)
            raw0 = None
            gridfile.write_bytes(q.spec(gr))
            prev = None
            if fault == "cap":
                args[args.index("--budget") + 1] = "1"
            if fault == "nan-tail":
                pp = wd / "tail.f32"
                data = pr.read_bytes()
                pp.write_bytes(data[:-4] + struct.pack("<f", float("nan")))
                args[args.index("--input") + 1] = str(pp)
            if fault == "exists":
                raw0 = b"protected previous file"
                raw.write_bytes(raw0)
            if fault == "alias":
                args[args.index("--output") + 1] = str(gridfile)
                prev = sha(gridfile)
            if fault == "extra":
                args += ["--batch", "8"]
            if fault == "bad-box":
                gridfile.write_bytes(b"CFDK-GRIDS-1\n63 18.4 0,0,62,1,1,2\n")
            if fault == "empty":
                gridfile.write_bytes(b"CFDK-GRIDS-1\n")
            if fault == "nonfinite-pitch":
                gridfile.write_bytes(b"CFDK-GRIDS-1\n63 nan 0,0,0,1,1,1\n")
            if fault == "seventeenth":
                gridfile.write_bytes(b"CFDK-GRIDS-1\n" + b"63 18.4 0,0,0,1,1,1\n" * 17)
            p = subprocess.run(args, capture_output=True, timeout=40)
            why = p.stderr.decode(errors="replace").strip()
            need(
                p.returncode != 0
                and msg in why
                and not p.stdout
                and not list(wd.glob("trial.mg.partial.*")),
                "wrong refusal: " + fault + " " + why,
            )
            if raw0:
                need(raw.read_bytes() == raw0, "existing output destroyed")
                raw.unlink()
            else:
                need(not raw.exists(), "published refused result")
            if prev:
                need(sha(gridfile) == prev, "source changed")
            refusals.append(dict(fault=fault, exit=p.returncode, reason=why))
        doc = dict(
            format="CFDK-GB10-multigrid-guards-v1",
            profile=s.PROFILE,
            build=q.BUILD,
            issuer=sha(s.__file__),
            script=sha(__file__),
            analytic=cases,
            sanitizers=tools,
            refusals=refusals,
            scope="Three signed synthetic requests, four standalone direct references per request, including same box on different coordinate grids. No synthetic specimen counted as physical.",
        )
        (stage / "guards.json").write_text(json.dumps(doc, indent=2) + "\n")
        publish(stage, dest)
        print(sha(dest / "guards.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    run(p.parse_args().out)
