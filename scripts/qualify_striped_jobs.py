"""Qualify streaming summary for four witnessed bridge jobs and the incomplete-coverage control."""

import argparse
import json
from pathlib import Path

from compare_native import sha
from qualify_anchors import oracle, unpack
from remount_loss import write_doc
from stripe_review import review
from validate_roi import require


def qualify(reports, bridge_jobs, legacy_edge, legacy_native, out):
    require(not out.exists(), "fresh evidence required")
    study = json.loads((reports / "bridge-result.json").read_bytes())["cases"]
    old = json.loads((reports / "region-qualification.json").read_bytes())["cases"]
    packs = json.loads((reports / "packed-edge-qualification.json").read_bytes())["cases"]
    cases = {}
    maps = 0
    cells = 0
    for n in [*study, "invalid-edge-control"]:
        legacy = n not in study
        conf = old[n] if legacy else study[n]
        parent = legacy_edge / n if legacy else bridge_jobs / n / "edge"
        root = legacy_native / n if legacy else bridge_jobs / n
        pin = packs[n]["packed_edge_sha256"] if legacy else conf["edge_sha256"]
        pts = [[11, 223], [12, 218]] if legacy else [[11, 46], [29, 46]]
        records = {tuple(s["pair"]): s for s in conf["cases" if legacy else "rows"]}
        first = None
        trials = []
        for band in (1, 7, 32):
            d = review(parent, pin, pts, band)
            require(first is None or first == d["result"], "partition-dependent summary")
            first = d["result"]
            trials.append(d["work"])
        for s in first["scenarios"]:
            i, j = s["pair"]
            p = root / f"native-{i}-{j}.json"
            lab = unpack(p, records[i, j]["native_report_sha256" if legacy else "native_sha256"])
            raw = json.loads(p.read_bytes())
            state, _, _ = oracle(lab, *pts)
            obs = raw["coverage"]
            counts = [r["cells"] for r in raw["regions"]]
            require(
                (len(counts), max(counts, default=0), obs["flagged"], obs["unknown"], state)
                == (
                    s["observed_components"],
                    s["largest_observed_cells"],
                    s["observed_flagged_cells"],
                    s["unknown_cells"],
                    s["anchor_state"],
                ),
                "stream vs native or independent graph differs",
            )
            require(
                s["anchor_cells"]
                == [
                    "flagged" if lab[y][x] > 0 else "below" if lab[y][x] == 0 else "unknown"
                    for y, x in pts
                ],
                "endpoint status",
            )
            maps += 1
            cells += len(lab) * len(lab[0])
        cases[n] = dict(result=first, trials=trials)
    require((maps, cells) == (40, 590400), "wrong scope")
    write_doc(
        out,
        dict(
            scope="Same pinned wall payloads in 1, 7 and 32-row bands; independent full native reports and material-space heap oracle. No physical accuracy claim.",
            source_sha256={
                n: sha(Path(__file__).with_name(n))
                for n in ("stripe_review.py", "stripe_frontier.py", "qualify_striped_jobs.py")
            },
            maps=maps,
            pair_cells=cells,
            cases=cases,
        ),
    )
    print("PASS: 3 partitions, 5 jobs, 40 native scenarios incl. incomplete coverage")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for s in ("reports", "bridge_jobs", "legacy_edge", "legacy_native", "out"):
        p.add_argument(s, type=Path)
    qualify(**vars(p.parse_args()))
