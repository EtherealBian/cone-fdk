"""Check a relocated stdlib atlas against independent NumPy/SciPy reads of authenticated GB10 parcels."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from spark_publication import fresh, publish, staging

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAIN = "fcaec11cafcbec9aae1cbab576252d5aee81b6b9b623e0a8caef56f7a0eda91c"
PACK = "595ca754f2c85e795021a8de8290e708f524054fd88c466a521bc223392fa6a7"
RULE = "f21f00ac180c0300b618feb2d5808615587ffe19529d8e0e54079a62db385890"


def check(ok):
    if not ok:
        raise ValueError("independent atlas gate failed")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bound(p, expected):
    data = p.read_bytes()
    check(digest(data) == expected)
    return data


def periodic_ids(m):
    # Label without wrap, then take connected components of the graph of labels meeting across the seam.
    z, n = label(m == 1)
    a, b = z[:, 0], z[:, -1]
    good = (a > 0) & (b > 0)
    src, dst = a[good], b[good]
    graph = coo_matrix((np.ones(len(src), dtype="u1"), (src, dst)), shape=(n + 1, n + 1))
    _, groups = connected_components(graph, directed=False)
    flat = groups[z.ravel()]
    fg = z.ravel() != 0
    mins = np.full(n + 1, z.size + 1, dtype="i4")
    np.minimum.at(mins, flat[fg], np.flatnonzero(fg) + 1)
    return np.where(fg, mins[flat], np.where(m.ravel() == 2, -1, 0)).reshape(m.shape)


def foot(mask, anchors):
    ids = periodic_ids(mask)
    u, counts = np.unique(ids[ids > 0], return_counts=True)
    a, b = (int(ids[tuple(t)]) for t in anchors)
    return dict(
        mask_sha256=digest(mask.astype("u1").tobytes()),
        cells=int(mask.sum()),
        components=np.stack([u, counts], axis=1).tolist(),
        anchor_ids=[a, b],
        anchors_linked=bool(a > 0 and a == b),
    )


def edge(x, y):
    pairs, counts = np.unique(
        np.stack([x.ravel(), y.ravel()], axis=1), axis=0, return_counts=True
    )
    flat = [(int(i), int(j), int(n)) for (i, j), n in zip(pairs, counts, strict=True)]
    idsx = set(x[x > 0].tolist())
    idsy = set(y[y > 0].tolist())
    o = sorted([i, j, n] for i, j, n in flat if i > 0 and j > 0)
    lost = sorted([i, 2 if j < 0 else 0, n] for i, j, n in flat if i > 0 and j <= 0)
    gain = sorted([j, 2 if i < 0 else 0, n] for i, j, n in flat if j > 0 and i <= 0)
    matrix = np.zeros((3, 3), dtype="i8")
    np.add.at(matrix, (np.where(x < 0, 2, x > 0).ravel(), np.where(y < 0, 2, y > 0).ravel()), 1)
    a, b = [np.asarray([row[d] for row in o], dtype="i4") for d in (0, 1)]

    def split(v):
        u, c = np.unique(v, return_counts=True)
        return u[c > 1].tolist()

    return dict(
        support_matrix=matrix.tolist(),
        overlaps=o,
        from_flagged_to_other=lost,
        from_other_to_flagged=gain,
        split_source_ids=split(a),
        merged_target_ids=split(b),
        no_overlap_source_ids=sorted(idsx - set(a.tolist())),
        no_overlap_target_ids=sorted(idsy - set(b.tolist())),
    )


def compare(report, source):
    chain = json.loads(bound(source / "chain.json", CHAIN))
    survey = json.loads(bound(source / "parcels/result.json", chain["spark_survey_sha256"]))
    expect = {
        k: v["reason"] for k, v in survey["cases"].items() if v["status"] == "upstream_refused"
    }
    check(
        report["upstream_refused"] == expect
        and set(report["cases"]) == set(survey["cases"]) - set(expect)
        and report["source_chain_sha256"] == CHAIN
        and report["protocol_sha256"] == RULE
    )
    totals = dict(
        cases=0,
        maps=0,
        cells=0,
        neighbor_comparisons=0,
        paired_cells=0,
        footprints=0,
        point_scenarios=0,
        split_steps=0,
        merge_steps=0,
        withheld=len(expect),
    )
    for case, c in report["cases"].items():
        rows = sorted((r for r in chain["rows"] if r["case"] == case), key=lambda r: r["pair"])
        keys = [r["pair"] for r in rows]
        check(
            c["ordered_pairs"] == keys
            and c["parcel_sha256_by_pair"] == [r["new_manifest_sha256"] for r in rows]
        )
        masks = []
        labs = []
        for r in rows:
            i, j = r["pair"]
            base = source / "parcels" / case / f"{i}-{j}"
            p = json.loads(bound(base / "manifest.json", r["new_manifest_sha256"]))
            check(all(p[k] == v for k, v in c["source"].items()) and p["pair"] == [i, j])
            shape = p["shape_ha"]
            anchors = p["anchors_row_col"]
            m = np.frombuffer(bound(base / "mask.u8", r["mask_sha256"]), "u1").reshape(shape)
            ids = periodic_ids(m)
            check(ids.astype("<i4").tobytes() == bound(base / "labels.i32", r["labels_sha256"]))
            masks.append(m)
            labs.append(ids)
        data = np.stack(masks)
        num = len(masks)
        yes = (data == 1).sum(axis=0)
        unknown = (data == 2).sum(axis=0)
        classes = [
            yes == num,
            (data == 0).all(axis=0),
            (unknown == 0) & (yes > 0) & (yes < num),
            (unknown > 0) & (unknown < num),
            unknown == num,
        ]
        names = [
            "always_flagged",
            "always_below",
            "fully_observed_mixed",
            "partially_observed",
            "always_unknown",
        ]
        check(dict(zip(names, [int(z.sum()) for z in classes], strict=True)) == c["partition"])
        vote = np.stack([yes, unknown], axis=-1).astype("u1")
        v, n = np.unique(vote.reshape(-1, 2), axis=0, return_counts=True)
        check(
            c["votes_sha256"] == digest(vote.tobytes())
            and c["vote_counts"] == np.column_stack([v, n]).tolist()
        )
        fp = {
            k: foot(mask, anchors)
            for k, mask in [
                ("all_flagged", (data == 1).all(axis=0)),
                ("any_flagged", (data == 1).any(axis=0)),
                ("any_flagged_or_unknown", (data != 0).any(axis=0)),
            ]
        }
        check(fp == c["footprints"])
        link = [
            pair
            for pair, ids in zip(keys, labs, strict=True)
            if ids[tuple(anchors[0])] > 0 and ids[tuple(anchors[0])] == ids[tuple(anchors[1])]
        ]
        check(
            link == c["observed_link_settings"]
            and c["union_link_without_single_setting"]
            == bool(not link and fp["any_flagged"]["anchors_linked"])
        )
        wanted = []
        for i, j in keys:
            for v in ([i + 1, j], [i, j + 1]):
                if v not in keys:
                    continue
                e = edge(labs[keys.index([i, j])], labs[keys.index(v)])
                wanted.append(dict(source=[i, j], target=v, **e))
                totals["split_steps"] += bool(e["split_source_ids"])
                totals["merge_steps"] += bool(e["merged_target_ids"])
        check(c["transitions"] == wanted)
        for point in c["points"]:
            y, x = point["row_col"]
            check(
                point
                == dict(
                    row_col=[y, x],
                    flagged_count=int(yes[y, x]),
                    unknown_count=int(unknown[y, x]),
                    labels_by_pair=[int(v[y, x]) for v in labs],
                )
            )
            totals["point_scenarios"] += num
        totals["cases"] += 1
        totals["maps"] += num
        totals["cells"] += data.size
        totals["neighbor_comparisons"] += len(wanted)
        totals["paired_cells"] += len(wanted) * masks[0].size
        totals["footprints"] += len(fp)
    check(
        totals["maps"] == report["maps"] == 67
        and totals["cells"] == report["scenario_cells"] == 988920
        and totals["neighbor_comparisons"] == 88
    )
    return totals


def run(source, bundle, job):
    job = fresh(job)
    check(digest((ROOT / "fixtures/threshold-atlas-protocol-v1.json").read_bytes()) == RULE)
    bound(bundle, PACK)
    with staging(job, ".threshold-atlas-") as tmp:
        stage = Path(tmp) / "accepted"
        stage.mkdir()
        with tempfile.TemporaryDirectory(dir=tmp) as sandbox:
            d = Path(sandbox)
            tools = ["offline_campaign.py", "threshold_atlas.py"]
            pack = d / "renamed.bin"
            shutil.copyfile(bundle, pack)
            for n in tools:
                shutil.copyfile(HERE / n, d / n)
            cli = [sys.executable, "-I", "-S", "-B", str(d / tools[1]), str(pack), CHAIN]
            raw = subprocess.check_output(cli, cwd=d)
            doc = json.loads(raw)
            summary = compare(doc, source)
            points = [
                (case, y, x)
                for case, rec in sorted(doc["cases"].items())
                for y, x in rec["source"]["anchors_row_col"]
            ]
            args = [s for case, y, x in points for s in ("--point", case, str(y), str(x))]
            probed = json.loads(subprocess.check_output(cli + args, cwd=d))
            probes = compare(probed, source)
            witness = {case: r.pop("points") for case, r in probed["cases"].items()}
            check(all(len(x) == 2 for x in witness.values()) and probes["point_scenarios"] == 134)
            for rec in doc["cases"].values():
                check(rec.pop("points") == [])
            check(doc == probed)
            bad = []
            for why, add, where in [
                ("unsubmitted", ["--point", "slide", "0", "0"], cli),
                ("out-of-grid", ["--point", "critical-a", "41", "0"], cli),
                ("wrong-parent", [], cli[:-1] + ["0" * 64]),
            ]:
                res = subprocess.run(where + add, cwd=d, capture_output=True)
                check(res.returncode != 0 and res.stdout == b"")
                bad.append([why, res.returncode])
            with pack.open("ab") as f:
                f.write(b"X")
            res = subprocess.run(cli, cwd=d, capture_output=True)
            check(res.returncode != 0 and not res.stdout)
            bad.append(["trailing-byte", res.returncode])
            check(set(p.name for p in d.iterdir()) == {"renamed.bin", *tools})
        (stage / "atlas.json").write_bytes(raw)
        (stage / "anchor-votes.json").write_text(
            json.dumps(witness, sort_keys=True, indent=2) + "\n"
        )
        result = dict(
            format="CFDK-threshold-atlas-qualification-v1",
            source_chain_sha256=CHAIN,
            pack_sha256=PACK,
            protocol_sha256=RULE,
            code={
                n: digest((HERE / n).read_bytes()) for n in [*tools, "qualify_threshold_atlas.py"]
            },
            atlas_sha256=digest(raw),
            anchor_votes_sha256=digest((stage / "anchor-votes.json").read_bytes()),
            numpy=np.__version__,
            scipy=scipy.__version__,
            oracle=summary,
            point_scenarios=probes["point_scenarios"],
            rejected_without_stdout=bad,
            scope="Eight simulated comparison cases and two upstream refusals; no measured specimen accuracy, no matching of physical objects.",
        )
        (stage / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
        publish(stage, job)
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("pack", type=Path)
    p.add_argument("job", type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.source, a.pack, a.job)))
