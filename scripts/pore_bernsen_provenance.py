"""Replay a period Fiji plugin on native CUDA-denoised byte planes, outside formal ROIs."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from pore_patch_stats import (
    DEV,
    ROOT,
    SETTINGS,
    Denoise,
    H,
    decision,
    median_stack,
    score,
    stack,
    verified,
)
from pore_slice_noise import md
from scipy import ndimage as ndi

PLUGIN = "17770eb4bbd9e2215a1c2aa28106d3e72578ce02c9db736429ff325e932b4166"
JAR = "ebdcfdaa663851d710958561e1179e770ecbb17ee5eab51f7e95187fd107b00b"
SEAL = ROOT / "pores/evidence/bernsen-decisions.json"


def hashfile(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def quantize(image, i):
    slope, shift, _ = SETTINGS[i]
    return np.clip(np.floor((image + shift) * slope + 0.5), 0, 255).astype("u1")


def rules(u, ct):
    if u.ndim != 3 or u.dtype != np.uint8:
        raise ValueError("Expected byte planes")
    y, x = np.ogrid[-5:6, -5:6]
    f = (x * x + y * y <= 26)[None]
    lo = ndi.minimum_filter(u, footprint=f).astype("i2")
    hi = ndi.maximum_filter(u, footprint=f).astype("i2")
    low = hi - lo < ct
    v = u.astype("i2")
    mid = (lo + hi) // 2
    return np.where(low, mid < 128, v < mid), np.where(low, v <= 128, 2 * v <= lo + hi), low


def predict(cache, out, plugin, jar, jdk):
    if hashfile(plugin) != PLUGIN or hashfile(jar) != JAR:
        raise ValueError("Wrong Fiji source or ImageJ jar")
    out.mkdir(parents=True, exist_ok=False)
    classes = out / "classes"
    classes.mkdir()
    src = out / "Auto_Local_Threshold.java"
    shutil.copyfile(plugin, src)
    c = str(jdk / ("javac.exe" if os.name == "nt" else "javac"))
    j = str(jdk / ("java.exe" if os.name == "nt" else "java"))
    adapter = ROOT / "scripts/BernsenProbe.java"
    subprocess.run(
        [
            c,
            "-J-Xmx96m",
            "-J-XX:ActiveProcessorCount=2",
            "-encoding",
            "UTF-8",
            "-cp",
            str(jar),
            "-d",
            str(classes),
            str(src),
            str(adapter),
        ],
        check=True,
    )
    rec = {}
    crop = (slice(None), slice(H, -H), slice(H, -H))
    with Denoise(ROOT / "build/pores/search-support.ptx") as gpu:
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            m = median_stack(raw, i, 176, 184, H)[:, :, 152 : 256 + 2 * H]
            den = gpu(m, 30 / SETTINGS[i][0])
            u = quantize(den, i)
            a, b, lo = rules(u, SETTINGS[i][2])
            a, b, lo = [v[crop] for v in (a, b, lo)]
            if not np.array_equal(a, decision(den, i, 0)[crop]):
                raise ValueError("Strict baseline differs")
            inp = out / f"{i}.u8"
            u.tofile(inp)
            f = out / f"{i}-plugin.u8"
            subprocess.run(
                [
                    j,
                    "-Xms8m",
                    "-XX:ActiveProcessorCount=2",
                    "-Xmx64m",
                    "-Djava.awt.headless=true",
                    "-cp",
                    os.pathsep.join([str(classes), str(jar)]),
                    "BernsenProbe",
                    str(inp),
                    str(f),
                    str(u.shape[2]),
                    str(u.shape[1]),
                    str(SETTINGS[i][2]),
                ],
                check=True,
            )
            binary = np.fromfile(f, "u1").reshape(u.shape)
            if not np.isin(binary, [0, 255]).all():
                raise ValueError("Plugin output is not binary")
            masks = dict(strict=a, plugin=binary[crop] == 0, printed=b, low=lo)
            np.savez_compressed(out / f"sample{i}.npz", **masks)
            rec[str(i)] = dict(
                raw=rs["sha256"],
                shape=list(u.shape),
                byte_sha256=hashfile(inp),
                plugin_out=hashfile(f),
                masks={k: md(v) for k, v in masks.items()},
            )
    return dict(
        plugin_source=PLUGIN,
        imagej_jar=JAR,
        adapter_sha256=hashlib.sha256(adapter.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        java=subprocess.check_output(
            [j, "-version"], stderr=subprocess.STDOUT, text=True
        ).strip(),
        javac=subprocess.check_output([c, "-version"], text=True).strip(),
        specimens=rec,
    )


def assess(masks, t):
    a, b, p, low = [masks[k] for k in ("strict", "printed", "plugin", "low")]
    old, new = score(a, t), score(b, t)
    flips = int(np.count_nonzero(a != p))
    transitions = {}
    for name, take in [("low", low), ("high", ~low)]:
        transitions[name] = {
            k: int(np.count_nonzero(take & v))
            for k, v in dict(
                unchanged_fp=a & b & ~t,
                unchanged_fn=~a & ~b & t,
                fp_added=~a & b & ~t,
                fp_fixed=a & ~b & ~t,
                fn_added=a & ~b & t,
                fn_fixed=~a & b & t,
            ).items()
        }
    advance = (
        flips == 0
        and new["dice"] - old["dice"] >= 0.005
        and all(new[k] - old[k] >= -0.005 for k in ("precision", "recall"))
    )
    return dict(
        strict=old,
        printed=new,
        plugin=score(p, t),
        implementation_disagreements=flips,
        printed_disagreements=int((a != b).sum()),
        transitions=transitions,
        advance=advance,
    )


def evaluate(cache, out):
    doc = json.loads((out / "predictions.json").read_text())
    results = {}
    if set(doc["specimens"]) != set("235") or doc != json.loads(SEAL.read_text()):
        raise ValueError("Expected sealed development predictions")
    for i in DEV:
        with np.load(out / f"sample{i}.npz") as f:
            m = {k: f[k] for k in f}
        if {k: md(v) for k, v in m.items()} != doc["specimens"][str(i)]["masks"]:
            raise ValueError("Masks changed")
        seg, ss = verified(cache, i, "segmented", False)
        t = stack(seg, i, "segmented", 176, 184)[:, :, 152:]
        results[str(i)] = dict(segmented=ss["sha256"], **assess(m, t))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["predict", "score"])
    for k in ("cache", "out", "plugin", "jar", "jdk"):
        ap.add_argument("--" + k, type=Path, required=k in ("cache", "out"))
    a = ap.parse_args()
    pred = a.phase == "predict"
    if pred and not all((a.plugin, a.jar, a.jdk)):
        ap.error("Predict requires --plugin --jar --jdk (bin directory)")
    r = predict(a.cache, a.out, a.plugin, a.jar, a.jdk) if pred else evaluate(a.cache, a.out)
    with (a.out / ("predictions.json" if pred else "results.json")).open(
        "x", encoding="utf8"
    ) as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
