"""Bind independent region replays to the full native/edge regression, not a pilot scan."""

import argparse
import json
from pathlib import Path

from check_edge_regression import verify as check_edge
from compare_native import sha
from remount_loss import write_doc
from validate_roi import require


def verify(full, edge_replay, region_replay, baseline):
    parent = check_edge(full, edge_replay, baseline)
    require(parent["pytest"]["tests"] >= 327, "region tests missing")
    names = ("region-noisy-change.json", "region-qualification.json")
    hashes = {n: sha(full / n) for n in names}
    require(
        all(sha(region_replay / n) == h for n, h in hashes.items()),
        "fresh/archived region mismatch",
    )
    q = json.loads((full / names[1]).read_bytes())
    edges = json.loads((full / "edge-qualification.json").read_bytes())["cases"]
    require(
        q["native_sha256"]
        == {
            "cone-fdk.exe": parent["build"]["executable_sha256"],
            "kernels.ptx": parent["build"]["ptx_sha256"],
        },
        "different CUDA tools",
    )
    require(
        q["cases"]["noisy-change"]["region_sha256"] == hashes[names[0]]
        and all(
            c["edge_sha256"] == edges[n]["review_manifest_sha256"] for n, c in q["cases"].items()
        ),
        "unbound source or region result",
    )
    return dict(
        protocol="Full clean rebuild and non-CoreOnly acceptance. Global region outputs agree byte-for-byte with a separate qualification from archived edge jobs. Prior volumes and witness/edge campaign digests checked by the edge regression gate.",
        gpu=parent["gpu"],
        pytest=parent["pytest"],
        build=parent["build"],
        frozen_volumes_sha256=parent["frozen_volumes_sha256"],
        edge_replay_sha256=parent["edge_replay_sha256"],
        region_replay_sha256=hashes,
        native_pair_replays=sum(len(x["cases"]) for x in q["cases"].values()),
        pair_cells=sum(x["pair_cells"] for x in q["cases"].values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("full", "edge_replay", "region_replay", "baseline", "out"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    write_doc(a.out, verify(a.full, a.edge_replay, a.region_replay, a.baseline))
    print("PASS: full region regression and archived replay")
