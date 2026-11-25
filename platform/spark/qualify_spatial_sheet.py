"""Independent offline publication check: original parcels -> index pixels, and real librsvg display -> colored cell centers."""

import argparse
import base64
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
from spark_publication import fresh, publish, staging

HERE = Path(__file__).resolve().parent
CHAIN = "fcaec11cafcbec9aae1cbab576252d5aee81b6b9b623e0a8caef56f7a0eda91c"
PACK = "595ca754f2c85e795021a8de8290e708f524054fd88c466a521bc223392fa6a7"
ATLAS = "99debd530f6c431f5ae4a7a2bda91390f9a94392a01a49edc59a4ea541bfe601"
VERIFIERS = {
    "threshold_atlas.py": "e7037bd30642430d3f3cb50d7c4617bc692ecfe98212430712eda3ac414ba70e",
    "offline_campaign.py": "ca095a965c053210e03ed369c798f9dccc31bce968509f28b7c68d13fcbc7ece",
}
SVG = "{http://www.w3.org/2000/svg}"
XL = "{http://www.w3.org/1999/xlink}href"
# Only input: internally generated, already parsed/checked, pinned SVG bytes. Capture raster from installed native system GdkPixbuf/librsvg; pass pixels to the independent NumPy oracle.
RASTER = """import sys,gi,hashlib
b=sys.stdin.buffer.read(1048577);assert len(b)<=1048576 and hashlib.sha256(b).hexdigest()==sys.argv[1];gi.require_version('GdkPixbuf','2.0');from gi.repository import GdkPixbuf
loader=GdkPixbuf.PixbufLoader.new_with_type('svg');loader.write(b);loader.close();image=loader.get_pixbuf();assert image.get_width()==1200 and image.get_height()==720 and image.get_n_channels()==4 and image.get_bits_per_sample()==8 and image.get_rowstride()==4800;sys.stdout.buffer.write(image.get_pixels())"""


def verify(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def checked(p, k):
    b = p.read_bytes()
    verify(sha(b) == k, "source hash: " + str(p))
    return b


def palette(f, u, n):
    out = np.zeros((*f.shape, 3), "u1")
    vals = [
        ((f == 0) & (u == 0), "dfe8e6"),
        ((f > 0) & (f < n) & (u == 0), "d9a125"),
        (f == n, "b53d36"),
        ((u > 0) & (u < n) & (f == 0), "92a3bb"),
        ((u > 0) & (f > 0), "6c508c"),
        (u == n, "343d53"),
    ]
    covers = np.zeros(f.shape, "u1")
    for bit, h in vals:
        out[bit] = list(bytes.fromhex(h))
        covers += bit
    verify((covers == 1).all(), "overlapping or missing oracle color")
    return out


def inspect(svg):
    verify(
        len(svg) <= 1 << 20 and not any(t in svg for t in (b"<!", b"<?", b"url(", b"@import")),
        "unexpected markup",
    )
    doc = ET.fromstring(svg)
    tags = {"svg", "title", "desc", "metadata", "rect", "g", "text", "image", "path"}
    attrs = {
        "width",
        "height",
        "viewBox",
        "role",
        "aria-labelledby",
        "id",
        "fill",
        "font-family",
        "font-size",
        "font-weight",
        "x",
        "y",
        "style",
        "preserveAspectRatio",
        "d",
        "stroke",
        "stroke-width",
        XL,
    }
    for e in doc.iter():
        verify(
            e.tag.startswith(SVG) and e.tag[len(SVG) :] in tags and set(e.attrib) <= attrs,
            "unexpected sheet content",
        )
        if "style" in e.attrib:
            verify(
                e.tag == SVG + "image" and e.get("style") == "image-rendering:pixelated",
                "unexpected style",
            )
        if XL in e.attrib:
            verify(
                e.tag == SVG + "image" and e.get(XL).startswith("data:image/png;base64,"),
                "external reference",
            )
    nodes = doc.findall(".//" + SVG + "image")
    verify(
        [e.get("id") for e in nodes] == ["full", "detail"], "expected exactly two inline rasters"
    )
    return json.loads(doc.find(SVG + "metadata").text), nodes


def compare(svg, atlas, case, rows, source, roi):
    md, nodes = inspect(svg)
    c = atlas["cases"][case]
    h, w = c["shape_ha"]
    y, x, rh, rw = roi
    keys = []
    masks = []
    n = len(rows)
    for r in sorted(rows, key=lambda t: t["pair"]):
        i, j = r["pair"]
        p = source / "parcels" / case / f"{i}-{j}"
        manifest = json.loads(checked(p / "manifest.json", r["new_manifest_sha256"]))
        verify(manifest["pair"] == [i, j] and manifest["shape_ha"] == [h, w], "wrong parcel")
        masks.append(np.frombuffer(checked(p / "mask.u8", r["mask_sha256"]), "u1").reshape(h, w))
        keys.append([i, j])
    m = np.stack(masks)
    f = np.sum(m == 1, axis=0, dtype="u1")
    u = np.sum(m == 2, axis=0, dtype="u1")
    z = f * 10 + u
    rgb = palette(f, u, n)
    yy = slice(y, y + rh)
    xx = np.arange(x, x + rw) % w
    cut = z[yy][:, xx]
    cutrgb = rgb[yy][:, xx, :]
    votes = sha(np.stack([f, u], axis=2).tobytes())
    verify(
        md["format"] == "CFDK-spatial-vote-sheet-v1"
        and md["case"] == case
        and md["source_chain_sha256"] == CHAIN
        and md["atlas_sha256"] == ATLAS
        and md["votes_sha256"] == votes == c["votes_sha256"]
        and md["roi_values_sha256"] == sha(cut.tobytes())
        and md["shape_row_col"] == [h, w]
        and md["settings"] == n
        and md["pairs"] == keys
        and md["roi_row_col_height_width"] == list(roi)
        and md["roi_columns_wrap"] == (x + rw > w)
        and md["vote_histogram"] == c["vote_counts"]
        and md["partition"] == c["partition"]
        and md["anchors_row_col"] == c["source"]["anchors_row_col"]
        and md["observed_link_settings"] == c["observed_link_settings"],
        "metadata or orientation",
    )
    data = subprocess.run(
        ["/usr/bin/python3", "-I", "-B", "-c", RASTER, sha(svg)],
        input=svg,
        capture_output=True,
        check=True,
        timeout=20,
    ).stdout
    display = np.frombuffer(data, "u1").reshape(720, 1200, 4)
    counts = []
    for node, ref, color in zip(nodes, (z, cut), (rgb, cutrgb), strict=True):
        b = base64.b64decode(node.get(XL).split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(b)) as img:
            verify(
                img.format == "PNG"
                and img.mode == "P"
                and list(ref.shape[::-1]) == list(img.size)
                and np.array_equal(np.asarray(img), ref)
                and np.array_equal(np.asarray(img.convert("RGB")), color),
                "independent index/palette mismatch",
            )
        ih, iw = ref.shape
        s = float(node.get("width")) / iw
        verify(
            s == float(node.get("height")) / ih and int(s) == s and s >= 1,
            "qualified native sheet scale",
        )
        sc = int(s)
        xp = int(node.get("x")) + np.arange(iw) * sc + (sc - 1) // 2
        yp = int(node.get("y")) + np.arange(ih) * sc + (sc - 1) // 2
        sample = display[np.ix_(yp, xp)]
        verify(
            np.all(sample[:, :, 3] == 255) and np.array_equal(sample[:, :, :3], color),
            "native SVG engine displayed wrong pixels",
        )
        counts.append(ref.size)
    return {
        "case": case,
        "settings": n,
        "full_pixels": counts[0],
        "roi_pixels": counts[1],
        "roi": list(roi),
        "sheet_sha256": sha(svg),
        "full_display_rgba_sha256": sha(data),
        "vote_sha256": votes,
    }


def run(pack, source, atlas_file, out):
    out = fresh(out)
    checked(pack, PACK)
    chain = json.loads(checked(source / "chain.json", CHAIN))
    atlas = json.loads(checked(atlas_file, ATLAS))
    codes = {**VERIFIERS, "spatial_sheet.py": sha((HERE / "spatial_sheet.py").read_bytes())}
    verified = []
    neg = []
    with staging(out, ".spatial-qual-") as tmp, tempfile.TemporaryDirectory() as t:
        stage = Path(tmp) / "accepted"
        stage.mkdir()
        cold = Path(t)
        for name, k in codes.items():
            (cold / name).write_bytes(checked(HERE / name, k))

        def invoke(case, extra=(), pin=CHAIN):
            return subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(cold / "spatial_sheet.py"),
                    str(pack.absolute()),
                    pin,
                    case,
                    *map(str, extra),
                ],
                capture_output=True,
                cwd=cold,
                timeout=60,
            )

        for case in sorted(atlas["cases"]):
            res = invoke(case)
            verify(res.returncode == 0, res.stderr[:250])
            rows = [r for r in chain["rows"] if r["case"] == case]
            v = compare(res.stdout, atlas, case, rows, source, (13, 348, 15, 24))
            (stage / (case + ".svg")).write_bytes(res.stdout)
            verified.append(v)
        case = "invalid-edge-control"
        roi = (8, 211, 9, 17)
        res = invoke(case, ["--roi", *roi])
        verify(res.returncode == 0, "custom ROI")
        v = compare(
            res.stdout, atlas, case, [r for r in chain["rows"] if r["case"] == case], source, roi
        )
        (stage / "invalid-edge-detail.svg").write_bytes(res.stdout)
        verified.append(v)
        for name, extra, pin in [
            ("slide", (), CHAIN),
            ("missing-reference", (), CHAIN),
            ("open-control", ["--roi", 40, 2, 2, 1], CHAIN),
            ("open-control", (), "0" * 64),
        ]:
            proc = invoke(name, extra, pin)
            verify(proc.returncode != 0 and not proc.stdout, "false output on refusal")
            neg.append(
                dict(case=name, args=list(extra), exit=proc.returncode, wrong_root=pin != CHAIN)
            )
        q = dict(
            format="CFDK-linearized-threshold-sheet-qualification-v1",
            chain_sha256=CHAIN,
            pack_sha256=PACK,
            atlas_sha256=ATLAS,
            code_sha256=codes,
            harness_sha256=sha(Path(__file__).read_bytes()),
            oracle="Unpacked authenticated masks + NumPy; Pillow verifies exact indices/colors; native GdkPixbuf/librsvg checks every displayed cell center at 1200x720. No interpolated screenshot used as data.",
            pillow=PIL.__version__,
            numpy=np.__version__,
            cases=len(atlas["cases"]),
            sheets=len(verified),
            verified=verified,
            refusals=neg,
            scope="Only the eight simulated comparison cases; no physical defect counts or calibrated sizes. No network, script or external reference in a qualified SVG.",
        )
        (stage / "qualification.json").write_text(json.dumps(q, indent=2) + "\n")
        publish(stage, out)
    print(sha((out / "qualification.json").read_bytes()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pack", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("atlas", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.pack, a.source, a.atlas, a.out)
