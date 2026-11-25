"""Create a deterministic, content-deduplicated detached evidence pack on Spark."""

import argparse
import os
import stat
import struct
import tempfile
from pathlib import Path

import offline_campaign as oc
from sealed_inputs import syncdir
from spark_publication import anchored


def source(path, root, expected, limit):
    # Producer reads trusted local tree; bytes, not filenames, must match the external pin.
    oc.need(
        path == path.resolve(strict=True) and path.is_relative_to(root),
        "source path escapes or aliases evidence",
    )
    fd = os.open(path, os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK | os.O_RDONLY)
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(fd)
        oc.need(stat.S_ISREG(st.st_mode) and 0 < st.st_size <= limit, "source type or size")
        data = f.read(st.st_size)
        oc.need(
            len(data) == st.st_size and not f.read(1) and oc.sha(data) == oc.pin(expected),
            "source digest or length",
        )
        return data


def write(stream, root, chain_pin):
    seen = set()
    stream.write(oc.MAGIC)

    def frame(path, pin, limit):
        data = source(
            path, root, pin, limit
        )  # even duplicates must exist and match at their claimed paths
        if pin not in seen:
            oc.need(stream.tell() + len(data) + 4 <= oc.MAX_PACK, "wire budget")
            stream.write(struct.pack("<I", len(data)))
            stream.write(data)
            seen.add(pin)
        return data

    chain = oc.obj(frame(root / "chain.json", chain_pin, oc.MAX_DOC))
    survey = oc.obj(frame(root / "parcels/result.json", chain["spark_survey_sha256"], oc.MAX_DOC))
    oc.plan(chain, survey)
    for row in chain["rows"]:
        case, a, b = oc.ident(row)
        n = f"{a}-{b}"
        parcel = root / "parcels" / case / n
        face = root / "faces" / case / n
        files = [
            parcel / "manifest.json",
            face / "manifest.json",
            root / "roi" / case / (n + ".json"),
            parcel / "mask.u8",
            parcel / "labels.i32",
            face / "face.u8",
        ]
        limits = [oc.MAX_DOC] * 3 + [oc.MAX_CELLS, 4 * oc.MAX_CELLS, oc.MAX_CELLS]
        for path, slot, limit in zip(files, oc.SLOTS, limits, strict=True):
            frame(path, row[slot], limit)


def pack(root, dest, chain_pin):
    root = Path(root).absolute()
    dest = Path(dest).absolute()
    oc.need(
        root == root.resolve(strict=True) and root.is_dir() and not dest.is_relative_to(root),
        "separate source and destination required",
    )
    with anchored(dest) as bound:
        with tempfile.NamedTemporaryFile(dir=bound.parent, prefix=".campaign-") as f:
            write(f, root, chain_pin)
            f.flush()
            os.fsync(f.fileno())
            result = oc.review(f.name, chain_pin)
            os.link(f.name, bound)
            syncdir(bound.parent)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("chain_sha256")
    a = p.parse_args()
    out = pack(a.source, a.output, a.chain_sha256)
    print(oc.encoded({k: v for k, v in out.items() if k != "rows"}).decode())
