"""Run the normal witness/edge/region qualification plus a fresh packed-backend comparison."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from validate_edges import qualify as edges
from validate_packed_edges import qualify as packed
from validate_regions import qualify as regions
from validate_witness import qualify as witnesses


def replay(trusted, reports, evidence):
    if evidence.exists():
        raise FileExistsError(evidence)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        # In a source checkout reports can also be the output directory. Freeze the references
        # before writing even the first fresh result, or an accidental change could self-qualify.
        frozen = root / "frozen"
        frozen.mkdir()
        for name in (
            "edge-qualification.json",
            "wall-pack-build.json",
            "body-witness.json",
            "region-qualification.json",
        ):
            shutil.copyfile(trusted / name, frozen / name)
        work = root / "witnesses"
        work.mkdir()
        witnesses(work, reports)
        edges(work, work / "edge-reviews", reports)
        regions(
            work / "edge-reviews",
            work / "region-reviews",
            reports,
            frozen / "edge-qualification.json",
        )
        packed(work, work / "edge-reviews", frozen, work / "packed")
        # The pinned packed qualification produced a complete file. Publish only after all gates.
        with TemporaryDirectory(dir=evidence.parent) as staging:
            partial = Path(staging) / "qualification.json"
            shutil.copyfile(work / "packed/qualification.json", partial)
            partial.rename(evidence)  # Windows same-volume publish; no replace.


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("trusted", "reports", "evidence"):
        p.add_argument(n, type=Path)
    replay(**vars(p.parse_args()))
