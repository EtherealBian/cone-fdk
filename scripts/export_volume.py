"""MetaImage header for physical coordinates, plus a portable axial slice contact viewer."""

import argparse
import base64
import html
import io
import json
from math import isfinite, prod
from pathlib import Path
from string import Template

import numpy as np
from PIL import Image


def export(volume, n, fov, outdir, title="ConeFDK volume"):
    return _export(volume, n, fov / n, (0, 0, 0), (n, n, n), outdir, title, None)


def export_layout(
    volume, layout_path, outdir, title="ConeFDK / selected volume", *, source_note=""
):
    # Accept native stdout or scripts/reconstruct.py stdout (including PowerShell UTF-16).
    metrics = json.loads(layout_path.read_bytes())
    n = metrics["n"]
    origin, shape = metrics["roi_origin_xyz"], metrics["output_shape_xyz"]
    if (
        type(n) is not int
        or not 2 <= n <= 4096
        or len(origin) != 3
        or len(shape) != 3
        or any(type(i) is not int for i in [*origin, *shape])
        or any(o < 0 or k < 1 or o + k > n for o, k in zip(origin, shape))
        or metrics["output_bytes"] != 4 * prod(shape)
    ):
        raise ValueError("invalid native output layout")
    return _export(
        volume,
        n,
        metrics["voxel_spacing_mm"],
        origin,
        shape,
        outdir,
        title,
        layout_path,
        source_note,
    )


def _export(volume, n, spacing, origin, shape, outdir, title, layout_path, source_note=""):
    viewer = outdir / ("roi-viewer.html" if layout_path is not None else "walnut-viewer.html")
    header = volume.with_suffix(".mhd")
    for target in (viewer, header):
        if target.exists() and (
            target.samefile(volume) or (layout_path is not None and target.samefile(layout_path))
        ):
            raise ValueError("an export path aliases its input")
    nx, ny, nz = shape
    if not isfinite(spacing) or spacing <= 0 or volume.stat().st_size != 4 * nx * ny * nz:
        raise ValueError("invalid spacing or raw file length for the specified grid")
    vol = np.memmap(volume, dtype="<f4", mode="r", shape=(nz, ny, nx))
    offset = " ".join(str((i - (n - 1) / 2) * spacing) for i in origin)
    if volume.suffix.lower() == ".mhd":
        raise ValueError("use a separate raw file, not a .mhd path")
    header.write_text(
        f"ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\nTransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = {offset}\nElementSpacing = {spacing} {spacing} {spacing}\nDimSize = {nx} {ny} {nz}\nElementType = MET_FLOAT\nElementDataFile = {volume.name}\n",
        encoding="utf-8",
    )
    sz, sy, sx = (max(1, v // 96) for v in (nz, ny, nx))
    lo, hi = np.percentile(vol[::sz, ::sy, ::sx], [2, 99])
    hi = max(hi, lo + 1e-9)
    frames = []
    ids = np.unique(np.linspace(0, nz - 1, min(81, nz), dtype=int))
    outdir.mkdir(parents=True, exist_ok=True)
    for i in ids:
        buff = io.BytesIO()
        Image.fromarray(np.uint8(np.clip((vol[i] - lo) / (hi - lo), 0, 1) * 255)).save(
            buff, format="PNG"
        )
        frames.append("data:image/png;base64," + base64.b64encode(buff.getvalue()).decode())
    templ = """<!doctype html><html lang="en"><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ConeFDK / volume</title><style>html{color-scheme:dark;background:#131a1d;color:#e0e6e2;font:16px system-ui}main{max-width:820px;margin:4vh auto;padding:0 18px}small,label,output{font:14px monospace;color:#a9c1b8}h1{font-size:clamp(24px,4vw,42px);margin:.3em 0}figure{margin:1.5em 0;display:grid;justify-items:center;background:#000;border:1px solid #485c54}img{width:min(64vh,100%);height:auto}input{width:100%;accent-color:#abe5c7}footer{padding:1em 0;color:#a2aea7}</style><main><small>CONEFDK / RECONSTRUCTION / LOCAL VIEWER</small><h1>$title</h1><p>Axial reconstruction · fixed attenuation window</p><figure><img id="slice" alt="Reconstructed axial section"></figure><label for="scrub">Axial section</label> <output id="pos"></output><input id="scrub" aria-label="Axial section" type="range" min="0" step="1"><footer>$info<br>Display previews only. Quantitative values are in the float32 volume. $source</footer></main><script>const images=$frames,ids=$ids,s=document.getElementById('scrub');s.max=ids.length-1;s.value=Math.floor(ids.length/2);function show(){document.getElementById('slice').src=images[s.value];document.getElementById('pos').textContent=ids[s.value]+' / '+ids.at(-1)}s.oninput=show;show()</script></html>"""
    # Substitute once: placeholders must never be interpreted inside embedded base64.
    viewer.write_text(
        Template(templ).substitute(
            title=html.escape(title),
            info=f"{nx} × {ny} × {nz} · {spacing:.5g} mm voxels · [{lo:.4g}, {hi:.4g}] /mm · {len(ids)} preview slices",
            frames=json.dumps(frames),
            ids=json.dumps((ids + origin[2]).tolist()),
            source="Slice labels are Z indices on the parent grid. " + html.escape(source_note)
            if layout_path is not None
            else "Source: Alexander Meaney, University of Helsinki · CC BY 4.0 · doi:10.5281/zenodo.6986012.",
        ),
        encoding="utf8",
    )
    verify_viewer(viewer, nz, size_xy=(nx, ny), z0=origin[2])
    return viewer


def verify_viewer(path, n, *, size_xy=None, z0=0):
    text = path.read_text(encoding="utf8").split("const images=", 1)[1]
    decode = json.JSONDecoder().raw_decode
    frames, end = decode(text)
    if not text.startswith(",ids=", end):
        raise ValueError("invalid slice metadata")
    ids, end = decode(text[end + 5 :])
    if ids != (np.unique(np.linspace(0, n - 1, min(n, 81), dtype=int)) + z0).tolist() or len(
        frames
    ) != len(ids):
        raise ValueError("slice schedule mismatch")
    for item in frames:
        prefix = "data:image/png;base64,"
        if not item.startswith(prefix):
            raise ValueError("missing embedded PNG")
        raw = base64.b64decode(item[len(prefix) :], validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG" or im.size != (size_xy or (n, n)):
                raise ValueError("bad slice dimensions or format")
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
    return len(frames)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("volume", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--n", type=int)
    group.add_argument("--layout", type=Path, help="JSON from the matching native reconstruction")
    p.add_argument("--fov", type=float)
    p.add_argument(
        "--source-note", default="", help="optional dataset attribution for a --layout viewer"
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.layout and a.fov is None:
        export_layout(a.volume, a.layout, a.out, source_note=a.source_note)
    elif a.n and a.fov is not None and not a.source_note:
        export(a.volume, a.n, a.fov, a.out, "Inside a walnut")
    else:
        p.error("use either --layout, or both --n and --fov")
