"""From independent line integrals to a material-datum veto or a CPU-checked loss map."""

import argparse
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from common import REPORT, ROOT
from compare_native import sha
from loss_oracle import cpu_labels, unpack
from loss_phantom import MU, PHOTONS, RELIEF, RO, design
from remount_loss import run, write_doc
from remount_phantom import BALLS, pose, project
from validate_remount import FIXTURE, GATE, GRID, readwalls
from validate_roi import require
from witness_phantom import INSERTS, generate
from witness_probe import audit

BODY = ROOT / "fixtures/body-witness.json"


def pack_pivot(job, path):
    payload = {
        key: (job / key).read_bytes()
        for key in ["manifest.json", "fixture.json", "body-witness.json"]
    }
    scan = json.loads(payload["manifest.json"])["scans"]["inspected"]
    _, _, z0, n, _, depth = scan["scout_roi_xyz_nxyz"]
    scout = np.fromfile(job / "inspected/scout.f32", "<f4").reshape(depth, n, n)
    for rec in scan["body_witnesses"]:
        x, y, z, w, h, d = rec["parent_roi_xyz_nxyz"]
        payload[f"insert-{rec['id']}.f32"] = scout[
            z - z0 : z - z0 + d, y : y + h, x : x + w
        ].tobytes()
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in payload.items():
            entry = zipfile.ZipInfo(name)
            entry.create_system = 0
            entry.external_attr = 0x20
            archive.writestr(entry, raw, compress_type=zipfile.ZIP_DEFLATED)
    require(
        [r["status"] for r in audit(path)] == ["pass", "displaced", "displaced"],
        "offline pivot audit",
    )


def axial_margin(reference, base, dest):
    """A decision threshold is not a minimum detectable physical shift. Probe its vicinity."""
    rows = []
    base.mkdir(parents=True, exist_ok=False)
    for microns in (30, 50, 70):
        slide = microns / 1000
        scan = base / f"slide-{microns}"
        project(scan, "second", relief=False, rack=BALLS + [0, 0, slide], witnesses=INSERTS)
        job = base / f"job-{microns}"
        log = run(
            reference,
            scan,
            FIXTURE,
            GRID,
            job,
            level=MU / 2,
            threshold=GATE,
            radius=RO,
            witness=BODY,
        )
        info = log["scans"]["inspected"]
        rs = info["body_witnesses"]
        residuals = [r["residual_mm"] for r in rs]
        require(
            all(e is not None and abs(e - slide) <= 0.025 for e in residuals),
            "near-limit residual vs design",
        )
        limit = json.loads(BODY.read_bytes())["residual_limit_mm"]
        accepted = max(residuals) <= limit
        require(
            log["status"] == ("compared" if accepted else "datum_rejected"),
            "wrong near-limit decision",
        )
        if microns != 50:
            require(accepted == (microns == 30), "near-limit bracket changed")
        flagged = None
        if accepted:
            flagged = log["loss"]["coverage"]["flagged"]
            require(
                flagged == 0 and not log["loss"]["coverage"]["unknown"], "false loss near gate"
            )
        else:
            require(
                log["reason"] == "body_witness_failed"
                and log["rejected_scan"] == "inspected"
                and log["loss"] is None
                and not list(job.rglob("*.wall"))
                and not list(job.rglob("shells.f32"))
                and not (job / "material-loss.json").exists(),
                "rejected marginal shift reached metrology",
            )
        rows.append(
            dict(
                design_axial_shift_mm=slide,
                fixture_residual_mm=info["material_to_scanner"]["max_residual_mm"],
                witness_residuals_mm=residuals,
                status=log["status"],
                flagged=flagged,
                inspected_projection_sha256=log["inputs_sha256"]["inspected.f32"],
                manifest_sha256=sha(job / "manifest.json"),
            )
        )
    write_doc(
        dest / "witness-margin.json",
        dict(
            protocol="Three noiseless axial carrier shifts at one remount; not a detection probability or calibrated sensitivity.",
            reference_projection_sha256=sha(reference.with_suffix(".f32")),
            witness_limit_mm=limit,
            cases=rows,
        ),
    )


def qualify(base, dest, *, existing_scans=None):
    base.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    scans = existing_scans or base / "scans"
    if existing_scans is None:
        generate(scans)
    cases = {}
    logs = {}
    for name, ref, part, guarded in [
        ("repeat", "reference", "repeat", True),
        ("noisy-change", "noisy-reference", "noisy-changed", True),
        ("slide", "reference", "slide", True),
        ("pivot", "reference", "pivot", True),
        ("missing-reference", "missing", "repeat", True),
        ("slide-fixture-only", "reference", "slide", False),
        ("noisy-change-fixture-only", "noisy-reference", "noisy-changed", False),
    ]:
        job = base / name
        log = run(
            scans / ref,
            scans / part,
            FIXTURE,
            GRID,
            job,
            level=MU / 2,
            threshold=GATE,
            radius=RO,
            witness=BODY if guarded else None,
        )
        logs[name] = log
        entry = cases[name] = dict(
            status=log["status"],
            manifest_sha256=sha(job / "manifest.json"),
            scan_projection_sha256={
                k: log["inputs_sha256"][k + ".f32"] for k in ("reference", "inspected")
            },
            scans={},
        )
        for role, info in log["scans"].items():
            report = entry["scans"][role] = dict(
                fixture_residual_mm=info["material_to_scanner"]["max_residual_mm"],
                scout_projection_passes=info["scout_projection_passes"],
            )
            require(
                report["fixture_residual_mm"] <= 0.035 and report["scout_projection_passes"] == 1,
                "fixture or scout gate",
            )
            if not guarded:
                continue
            records = info["body_witnesses"]
            r, t = pose("first" if role == "reference" else "second")
            panel = []
            for point, record in zip(INSERTS, records, strict=True):
                observed = record["observed_scanner_xyz_mm"]
                # Distance to the known body center, independent of the carrier and its fitted pose.
                err = (
                    None
                    if observed is None
                    else float(np.linalg.norm(observed - (r @ point + t)))
                )
                require(err is None or err <= 0.035, "witness localization vs ray truth")
                panel.append(
                    dict(
                        id=record["id"],
                        status=record["status"],
                        residual_mm=record["residual_mm"],
                        truth_center_error_mm=err,
                    )
                )
            report["witnesses"] = panel
        if name in ("slide", "pivot", "missing-reference"):
            role = "reference" if name == "missing-reference" else "inspected"
            require(
                log["status"] == "datum_rejected"
                and log["reason"] == "body_witness_failed"
                and log["rejected_scan"] == role
                and log["loss"] is None
                and not list(job.rglob("*.wall"))
                and not list(job.rglob("shells.f32"))
                and not (job / "material-loss.json").exists(),
                "invalid datum reached measurement",
            )
            rs = log["scans"][role]["body_witnesses"]
            wanted = {
                "slide": ["displaced"] * 3,
                "pivot": ["pass", "displaced", "displaced"],
                "missing-reference": ["pass", "missing", "pass"],
            }
            require([x["status"] for x in rs] == wanted[name], "wrong witness veto")
            if name == "slide":
                require(all(abs(x["residual_mm"] - 0.8) < 0.035 for x in rs), "slide vs design")
            if name == "pivot":
                for p, x in zip(INSERTS, rs, strict=True):
                    expected = 2 * np.linalg.norm((p - INSERTS[0])[:2]) * np.sin(np.deg2rad(2))
                    require(abs(x["residual_mm"] - expected) < 0.035, "pivot vs design")
                pack_pivot(job, dest / "witness-pivot.zip")
            if name == "missing-reference":
                require(
                    rs[1]["residual_mm"] is None and not (job / "inspected").exists(),
                    "missing reference failed open",
                )
            entry.update(rejected_scan=role, reason=log["reason"], produced_wall_or_loss=False)
            continue
        require(log["status"] == "compared", "unexpected rejection: " + name)
        doc = json.loads((job / "material-loss.json").read_bytes())
        a, b = readwalls(job)
        labels = unpack(doc)
        np.testing.assert_array_equal(labels, cpu_labels(a, b, GATE))
        cov = doc["coverage"]
        require(cov["unknown"] == 0 and cov["paired_valid"] == 14760, "coverage")
        regs = doc["regions"]
        entry.update(
            coverage=cov, region_count=len(regs), loss_sha256=sha(job / "material-loss.json")
        )
        if name == "repeat":
            maxdelta = float(np.max(np.abs(a["thickness"] - b["thickness"])))
            require(not regs and maxdelta <= 0.06, "false change on remount")
            entry["max_abs_delta_mm"] = maxdelta
        elif name == "slide-fixture-only":
            require(
                regs and cov["flagged"] > 0,
                "fixture-only negative control did not expose false change",
            )
            entry["spurious_peak_mm"] = max(reg["peak"]["loss_mm"] for reg in regs)
        else:
            ideal = design(None, RELIEF)[1]
            truth = ideal >= GATE
            pred = labels > 0
            iou = float((pred & truth).sum() / (pred | truth).sum())
            pk = regs[0]["peak"] if len(regs) == 1 else {}
            require(
                len(regs) == 1
                and iou >= 0.90
                and abs(pk["loss_mm"] - ideal.max()) <= 0.05
                and abs(pk["azimuth_deg"] - RELIEF[0]) <= 3
                and abs(pk["outer_recession_mm"]) < 0.025,
                "relief metrology",
            )
            entry.update(
                design_peak_mm=float(ideal.max()),
                peak_mm=pk["loss_mm"],
                footprint_iou=iou,
                peak_azimuth_deg=pk["azimuth_deg"],
            )
    # Same acquisitions with witness on vs off: no extra reads, no adjustment of pose or loss.
    for name in ["noisy-change", "slide"]:
        a = logs[name]
        b = logs[name + "-fixture-only"]
        for role in ("reference", "inspected"):
            require(
                a["scans"][role]["material_to_scanner"] == b["scans"][role]["material_to_scanner"]
                and a["scans"][role]["scout_projection_passes"]
                == b["scans"][role]["scout_projection_passes"],
                "guard changed fit or scout pass count",
            )
        if name == "noisy-change":
            require(
                all(a["outputs_sha256"][k] == v for k, v in b["outputs_sha256"].items())
                and all(
                    a["scans"][r]["shell_projection_passes"]
                    == b["scans"][r]["shell_projection_passes"]
                    == 1
                    for r in a["scans"]
                ),
                "accepted payload or projection pass regression",
            )
    summary = dict(
        protocol="Independent analytic finite-tube scans. Carrier-only slips; body inserts never used for fitting. No physical dimensions validated.",
        witness_config_sha256=sha(BODY),
        witness_limit_mm=0.050,
        loss_threshold_mm=GATE,
        photons_per_air_bin=PHOTONS,
        noise_seeds=dict(reference=99581, changed=39089),
        same_accepted_payloads_and_pass_counts=True,
        example_pivot_zip_sha256=sha(dest / "witness-pivot.zip"),
        cases=cases,
    )
    write_doc(dest / "body-witness.json", summary)
    axial_margin(scans / "reference", base / "axial-margin", dest)
    print("PASS: body witnesses / detector-to-loss and fail-closed controls", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep", type=Path)
    p.add_argument("--reports", type=Path, default=REPORT)
    args = p.parse_args()
    with TemporaryDirectory() as tmp:
        qualify(args.keep or Path(tmp), args.reports)
