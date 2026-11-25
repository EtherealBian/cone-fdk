"""Qualify tilt correction from fresh simulated detector data, including negative controls."""

import argparse
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from axis_gauge import run
from axis_phantom import CASES, MU, NOISY, PHOTONS, SEED, design, project
from axis_probe import angle, review
from bore import gauge
from common import BUILD, REPORT, ROOT
from compare_native import sha
from validate_roi import require
from validate_stack import save_json

RESIDUAL = 0.025
LIMITS = {
    "axis_angle_deg": 0.12,
    "axis_origin_mm": 0.015,
    "diameter_mm": 0.03,
    "eccentricity_mm": 0.018,
    "section_center_mm": 0.018,
    "paired_dimensions_mm": 0.02,
    "alternative_angle_deg": 0.01,
    "alternative_origin_mm": 0.001,
}


def evaluate(base, reports, scans):
    cases = {}
    values = {}
    for name, case in CASES.items():
        job = base / name
        scan = scans / name
        log = run(
            scan,
            ROOT / "fixtures/axis-coarse.grid",
            job,
            fov=32,
            threshold=MU / 2,
            axis_limit=RESIDUAL,
            bore_limit=RESIDUAL,
        )
        fit = json.loads((job / "axis.json").read_bytes())
        center, u, _, _, off = design(case)
        h = fit["rows"]
        info = log["stages"]["axis"]
        require(
            info["redzones_verified"]
            and info["bytes_read"] == info["passes"] * info["input_bytes"]
            and info["peak_owned_bytes"] <= info["budget_bytes"]
            and info["output_bytes"] == (job / "axis.json").stat().st_size,
            "axis accounting",
        )
        evidence = cases[name] = dict(
            design=dict(
                tilt_deg=case[0],
                heading_deg=case[1],
                axis_unit=u.tolist(),
                origin_xyz_mm=center.tolist(),
                outer_semiaxes_mm=case[3],
                bore_semiaxes_mm=case[4],
                eccentricity_mm=case[5],
                bore_offset_xyz_mm=off.tolist(),
            ),
            status=log["status"],
            axis=fit,
            projection_sha256=log["inputs_sha256"][".f32"],
            artifacts_sha256=log["outputs_sha256"],
            projection_passes={
                s: log["stages"][s]["projection_passes"]
                for s in ("coarse", "aligned")
                if s in log["stages"]
            },
        )
        coarse = gauge(
            job / "coarse.wall",
            job / "coarse.grid",
            base / (name + "-uncorrected.json"),
            RESIDUAL,
        )
        evidence["uncorrected_rows_measured"] = coarse["measured"]
        if name == "oval-outside":
            require(
                log["status"] == "axis_rejected"
                and fit["status"] == "noncylindrical"
                and fit["measurement"] is None
                and not (job / "aligned.grid").exists()
                and not (job / "sections.json").exists(),
                "oval outer must stop before second reconstruction",
            )
            continue
        m = fit["measurement"]
        cpu = review(job)
        evidence["cpu"] = cpu
        errs = dict(
            axis_angle_deg=angle(u, m["axis_unit"]),
            axis_origin_mm=float(np.linalg.norm(center - m["origin_xyz_mm"])),
            alternative_angle_deg=cpu["alternative_angle_delta_deg"],
            alternative_origin_mm=cpu["alternative_origin_delta_mm"],
        )
        require(all(v <= LIMITS[k] for k, v in errs.items()), f"{name}: {errs}")
        sec = json.loads((job / "sections.json").read_bytes())["rows"]
        evidence["aligned_rows_measured"] = log["stages"]["bore"]["measured"]
        evidence["errors"] = errs
        if name == "oval-bore":
            require(
                log["status"] == "sections_rejected"
                and len(sec) == h
                and all(r["status"] == "noncircular" and r["measurement"] is None for r in sec),
                "oval bore must reject every section",
            )
            continue
        require(
            log["status"] == "measured"
            and len(sec) == h
            and all(r["status"] == "measured" for r in sec),
            "a good row rejected: " + name,
        )
        require(
            coarse["measured"] == (h if name == "upright" else 0), "unexpected uncorrected result"
        )
        vals = []
        centers = []
        normal = np.asarray(m["axis_unit"])
        for r in sec:
            v = r["measurement"]
            vals.append(
                [v["inner"]["diameter_mm"], v["outer"]["diameter_mm"], v["eccentricity_mm"]]
            )
            plane = m["origin_xyz_mm"] + normal * r["axial_offset_mm"]
            for edge, basepoint in [("outer", center), ("inner", center + off)]:
                expected = basepoint + u * ((plane - basepoint) @ normal) / (u @ normal)
                centers.append(float(np.linalg.norm(v[edge]["center_xyz_mm"] - expected)))
        values[name] = np.array(vals)
        dev = abs(values[name] - [2 * case[4][0], 2 * case[3][0], case[5]])
        errs.update(
            diameter_mm=float(dev[:, :2].max()),
            eccentricity_mm=float(dev[:, 2].max()),
            section_center_mm=max(centers),
        )
        require(all(v <= LIMITS[k] for k, v in errs.items()), f"{name}: {errs}")
        print(name, errs, flush=True)
    paired = {
        s: float(abs(values[s] - values["upright"]).max()) for s in ("tilted-same-part", NOISY)
    }
    require(
        max(paired.values()) <= LIMITS["paired_dimensions_mm"]
        and len({v["projection_sha256"] for v in cases.values()}) == len(cases),
        "remount comparison failed",
    )
    pack = reports / "axis-review.zip"
    example = base / "tilted-second-part"
    with zipfile.ZipFile(pack, "w") as f:
        for member in [
            "coarse.grid",
            "coarse.wall",
            "axis.json",
            "aligned.grid",
            "aligned.wall",
            "sections.json",
            "manifest.json",
        ]:
            f.writestr(
                zipfile.ZipInfo(member),
                (example / member).read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    save_json(
        reports / "axis-gauge.json",
        dict(
            protocol="Finite analytic tubes and independently projected poses. One fixed-seed counting-noise case. No physical specimen or general rigid registration.",
            binary_sha256={n: sha(BUILD / n) for n in ("cone-fdk.exe", "kernels.ptx")},
            limits=LIMITS,
            peak_residual_limit_mm=RESIDUAL,
            threshold=MU / 2,
            noise=dict(case=NOISY, photons_per_air_bin=PHOTONS, seed=SEED),
            paired_max_change_mm=paired,
            review_sha256=sha(pack),
            cases=cases,
        ),
    )


def qualify(base, reports):
    reports.mkdir(parents=True, exist_ok=True)
    scans = base / "scans"
    scans.mkdir(parents=True, exist_ok=True)
    for n, c in CASES.items():
        project(scans / n, c, noisy=n == NOISY)
    evaluate(base, reports, scans)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep", type=Path)
    p.add_argument("--reports", type=Path, default=REPORT)
    a = p.parse_args()
    with TemporaryDirectory() as t:
        qualify(a.keep or Path(t), a.reports)
