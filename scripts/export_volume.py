"""MetaImage header for physical coordinates, plus a portable axial slice contact viewer."""

import argparse
import base64
import html
import io
import json
from pathlib import Path
from string import Template

import numpy as np
from PIL import Image


def export(volume, n, fov, outdir, title="ConeFDK volume"):
    if volume.stat().st_size != 4 * n**3:
        raise ValueError("raw file length does not match n")
    vol = np.memmap(volume, dtype="<f4", mode="r", shape=(n, n, n))
    spacing = fov / n
    origin = -fov / 2 + spacing / 2
    volume.with_suffix(".mhd").write_text(
        f"ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\nTransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = {origin} {origin} {origin}\nElementSpacing = {spacing} {spacing} {spacing}\nDimSize = {n} {n} {n}\nElementType = MET_FLOAT\nElementDataFile = {volume.name}\n"
    )
    step = max(1, n // 96)
    lo, hi = np.percentile(vol[::step, ::step, ::step], [2, 99])
    hi = max(hi, lo + 1e-9)
    frames = []
    ids = np.unique(np.linspace(0, n - 1, min(81, n), dtype=int))
    outdir.mkdir(parents=True, exist_ok=True)
    for i in ids:
        buff = io.BytesIO()
        Image.fromarray(np.uint8(np.clip((vol[i] - lo) / (hi - lo), 0, 1) * 255)).save(
            buff, format="PNG"
        )
        frames.append("data:image/png;base64," + base64.b64encode(buff.getvalue()).decode())
    templ = """<!doctype html><html lang="en"><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ConeFDK / volume</title><style>html{color-scheme:dark;background:#131a1d;color:#e0e6e2;font:16px system-ui}main{max-width:820px;margin:4vh auto;padding:0 18px}small,label,output{font:14px monospace;color:#a9c1b8}h1{font-size:clamp(24px,4vw,42px);margin:.3em 0}figure{margin:1.5em 0;display:grid;justify-items:center;background:#000;border:1px solid #485c54}img{width:min(64vh,100%);height:auto}input{width:100%;accent-color:#abe5c7}footer{padding:1em 0;color:#a2aea7}</style><main><small>CONEFDK / MEASURED DATA / LOCAL VIEWER</small><h1>$title</h1><p>Axial reconstruction · corrected detector center · fixed attenuation window</p><figure><img id="slice" alt="Reconstructed axial section"></figure><label for="scrub">Axial section</label> <output id="pos"></output><input id="scrub" aria-label="Axial section" type="range" min="0" step="1"><footer>$info<br>Display previews only. Quantitative values are in the float32 volume. Source: Alexander Meaney, University of Helsinki · CC BY 4.0 · doi:10.5281/zenodo.6986012.</footer></main><script>const images=$frames,ids=$ids,s=document.getElementById('scrub');s.max=ids.length-1;s.value=Math.floor(ids.length/2);function show(){document.getElementById('slice').src=images[s.value];document.getElementById('pos').textContent=ids[s.value]+' / '+ids.at(-1)}s.oninput=show;show()</script></html>"""
    # Substitute once: placeholders must never be interpreted inside embedded base64.
    viewer = outdir / "walnut-viewer.html"
    viewer.write_text(
        Template(templ).substitute(
            title=html.escape(title),
            info=f"{n}³ · {spacing:.5g} mm voxels · [{lo:.4g}, {hi:.4g}] /mm · {len(ids)} preview slices",
            frames=json.dumps(frames),
            ids=json.dumps(ids.tolist()),
        ),
        encoding="utf8",
    )
    verify_viewer(viewer, n)


def verify_viewer(path, n):
    text = path.read_text(encoding="utf8").split("const images=", 1)[1]
    decode = json.JSONDecoder().raw_decode
    frames, end = decode(text)
    if not text.startswith(",ids=", end):
        raise ValueError("invalid slice metadata")
    ids, end = decode(text[end + 5 :])
    if ids != np.unique(np.linspace(0, n - 1, min(n, 81), dtype=int)).tolist() or len(
        frames
    ) != len(ids):
        raise ValueError("slice schedule mismatch")
    for item in frames:
        prefix = "data:image/png;base64,"
        if not item.startswith(prefix):
            raise ValueError("missing embedded PNG")
        raw = base64.b64decode(item[len(prefix) :], validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG" or im.size != (n, n):
                raise ValueError("bad slice dimensions or format")
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
    return len(frames)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("volume", type=Path)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--fov", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    export(a.volume, a.n, a.fov, a.out, "Inside a walnut")
