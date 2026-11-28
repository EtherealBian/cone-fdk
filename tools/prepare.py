"""Convert the published POPI Elekta exposure frames to calibrated line integrals."""

import argparse
import hashlib
import struct
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def prepare(folder):
    hashes = {
        "projections.tgz": "fce00ce73dda43b665176b2803f6aaa9140a27f8e72267ee7574711838696c72",
        "geometry.xml": "58998d13201706b3735aba162cecb317dd70cb95e0990a35447cbfddd94e21c1",
        "FRAME.DBF": "e98c4142032b2d2f21fb2f31d76b3e3d2337365293659465cae54541a46c1cfc",
    }
    for name, digest in hashes.items():
        with (folder / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                raise ValueError(f"not the published POPI file: {name}")

    root = ET.parse(folder / "geometry.xml").getroot()
    views = root.findall("Projection")
    sid, sdd = (
        root.findtext(k)
        for k in ("SourceToIsocenterDistance", "SourceToDetectorDistance")
    )
    db = (folder / "FRAME.DBF").read_bytes()
    count, header, stride = struct.unpack_from("<IHH", db, 4)
    fields, offset = {}, 1
    for pos in range(32, header - 1, 32):
        size = db[pos + 16]
        fields[db[pos : pos + 11].rstrip(b"\0").decode()] = slice(offset, offset + size)
        offset += size

    # The Synergy HIS encodes the complement of transmission, not attenuation.
    # Match RTK's reserved codes 0 and 65535 (both map to the value for code 1).
    code = np.arange(65536, dtype=np.float64)
    code[[0, -1]] = 1
    lut = (-np.log1p(-code / 65536)).astype("<f4")
    with (
        tarfile.open(folder / "projections.tgz", "r|gz") as tar,
        (folder / "views.f32").open("wb") as out,
        (folder / "views.geom").open("w") as geom,
    ):
        geom.write(f"POPI1 {len(views)} 512 512 0.8 0.8 {sid} {sdd}\n")
        for i, (view, member) in enumerate(zip(views, tar, strict=True)):
            rec = db[header + i * stride : header + (i + 1) * stride]
            if (
                not member.isfile()
                or member.name != f"{int(rec[fields['DBID']]):08X}.his"
                or int(rec[fields["SEQ_NUM"]]) != i + 1
            ):
                raise ValueError("frame sequence does not match acquisition table")
            # 68-byte HIS header + 32 bytes of binary extension; version pinned above.
            pixels = np.frombuffer(
                tar.extractfile(member).read(), "<u2", offset=100
            ).reshape(512, 512)
            lut[pixels].tofile(out)
            geom.write(
                " ".join(
                    view.findtext(k)
                    for k in ("GantryAngle", "ProjectionOffsetX", "ProjectionOffsetY")
                )
                + "\n"
            )
    if len(views) != count:
        raise ValueError("geometry/frame count mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    prepare(parser.parse_args().folder)
