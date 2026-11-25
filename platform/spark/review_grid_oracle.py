"""Recompute grid agreement directly from two externally pinned CFDKMG01 streams. Pure Python, single-plane memory."""

import argparse, hashlib, json, math, os, re, stat, struct
from pathlib import Path


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def header(stream):
    pre = stream.read(12)
    need(len(pre) == 12, "truncated header")
    magic, k = struct.unpack("<8sI", pre)
    need(magic == b"CFDKMG01" and 1 <= k <= 16, "grid header")
    data = stream.read(36 * k)
    need(len(data) == 36 * k, "truncated table")
    rows = list(struct.iter_unpack("<Iff6I", data))
    size = 12 + 36 * k
    for n, f, p, x, y, z, a, b, c in rows:
        need(
            2 <= n <= 4096
            and math.isfinite(f)
            and math.isfinite(p)
            and f > 0
            and p > 0
            and struct.unpack("<f", struct.pack("<f", f / n))[0] == p
            and min(a, b, c) > 0
            and x + a <= n
            and y + b <= n
            and z + c <= n,
            "invalid spatial grid",
        )
        size += 4 * a * b * c
    st = os.fstat(stream.fileno())
    need(
        stat.S_ISREG(st.st_mode) and size == st.st_size and size <= (64 << 30) + 1024,
        "bad pack length",
    )
    return pre + data, rows


def calc(native, witness, npin, wpin):
    need(
        all(type(h) is str and re.fullmatch("[0-9a-f]{64}", h) for h in (npin, wpin)),
        "external digests required",
    )
    out = []
    with open(native, "rb") as a, open(witness, "rb") as b:
        ha, rows = header(a)
        hb, _ = header(b)
        need(ha == hb, "incompatible grid descriptors")
        hs = [hashlib.sha256(h) for h in (ha, hb)]
        for n, fov, p, x0, y0, z0, nx, ny, nz in rows:
            lines = []
            largest = -1
            point = None
            fmt = struct.Struct("<" + str(nx) + "f")
            cells = [hashlib.sha256(), hashlib.sha256()]
            for z in range(nz):
                sums = []
                peak = 0.0
                for iy in range(ny):
                    bufs = [i.read(fmt.size) for i in (a, b)]
                    need(all(len(v) == fmt.size for v in bufs), "incomplete paired plane")
                    for h, s, v in zip(hs, cells, bufs, strict=True):
                        h.update(v)
                        s.update(v)
                    v, r = map(fmt.unpack, bufs)
                    d = tuple(l - k for l, k in zip(v, r, strict=True))
                    need(all(map(math.isfinite, d)), "nonfinite delta")
                    rp = max(map(abs, d))
                    peak = max(peak, rp)
                    if rp > largest:
                        idx = next(i for i, t in enumerate(d) if abs(t) == rp)
                        largest = rp
                        point = [x0 + idx, y0 + iy, z0 + z]
                    sums.append(
                        (math.fsum(t * t for t in d), math.fsum(t * t for t in r), math.fsum(d))
                    )
                se, en, si = map(math.fsum, zip(*sums))
                lines.append(dict(z=z + z0, sse=se, energy=en, signed_sum=si, max_abs=peak))
            energy = math.fsum(s["energy"] for s in lines)
            need(energy > 0, "blank reference")
            count = nx * ny * nz
            out.append(
                dict(
                    fov_mm=fov,
                    pitch_mm=p,
                    parent_n=n,
                    box=[x0, y0, z0, nx, ny, nz],
                    voxel_pins=[h.hexdigest() for h in cells],
                    voxels=count,
                    rel_l2=math.sqrt(math.fsum(s["sse"] for s in lines) / energy),
                    signed_mean=math.fsum(s["signed_sum"] for s in lines) / count,
                    max_abs=largest,
                    max_parent_xyz=point,
                    planes=lines,
                )
            )
        need(
            not a.read(1) and not b.read(1) and [h.hexdigest() for h in hs] == [npin, wpin],
            "pack digest mismatch",
        )
    return out


def check(qualification, pin, native, witness):
    need(type(pin) is str and re.fullmatch("[0-9a-f]{64}", pin), "invalid external receipt pin")
    with Path(qualification).open("rb") as receipt:
        raw = receipt.read((2 << 20) + 1)
    need(
        len(raw) <= 2 << 20 and hashlib.sha256(raw).hexdigest() == pin, "receipt digest mismatch"
    )
    d = json.loads(raw)
    typ = d["format"]
    need(
        typ in ("CFDK-Spark-independent-grid-oracle-v1", "CFDK-Spark-user-grid-audit-v1")
        and d["passed"] is True,
        "invalid qualification",
    )
    npin = d["native_pack_sha256"]
    wpin = d["oracle_pack"]["pack_sha256"]
    out = calc(native, witness, npin, wpin)
    bounds = d["limits"]
    required = dict(native_rel_l2=0.002, native_max_abs=0.0002)
    if typ == "CFDK-Spark-independent-grid-oracle-v1":
        required.update(control_rel_l2=0.0001, control_max_abs=0.00002)
    need(bounds == required, "changed gates")
    close = (
        lambda a, b: type(b) in (int, float)
        and math.isfinite(b)
        and math.isclose(a, b, rel_tol=5e-14, abs_tol=1e-25)
    )
    offset = 12 + 36 * len(out)
    need(d["oracle_pack"]["format"] == "CFDKMG01", "incorrect pack format")
    for r, g, wp in zip(out, d["regions"], d["oracle_pack"]["regions"], strict=True):
        vox = r["voxels"]
        n = r["parent_n"]
        bb = r["box"]
        p = r["pitch_mm"]
        expected = dict(
            n=n,
            fov_mm=r["fov_mm"],
            pitch_mm=p,
            origin_xyz=bb[:3],
            shape_xyz=bb[3:],
            first_voxel_center_mm=[(a - (n - 1) / 2) * p for a in bb[:3]],
            offset_bytes=offset,
            size_bytes=4 * vox,
            sha256=r["voxel_pins"][1],
        )
        need(
            wp == expected and g["fov_mm"] == r["fov_mm"], "recorded reference descriptor differs"
        )
        offset += 4 * vox
        t = g["native_vs_astra"]
        need(
            r["parent_n"] == g["parent_n"]
            and r["box"] == g["box"]
            and r["voxel_pins"][1] == wp["sha256"]
            and r["rel_l2"] <= bounds["native_rel_l2"]
            and r["max_abs"] <= bounds["native_max_abs"],
            "region identity or gate",
        )
        need(
            all(r[k] == t[k] for k in ("voxels", "max_abs", "max_parent_xyz"))
            and all(close(r[k], t[k]) for k in ("rel_l2", "signed_mean")),
            "global metric disagreement",
        )
        for u, v in zip(r["planes"], t["planes"], strict=True):
            need(
                u["z"] == v["z"]
                and u["max_abs"] == v["max_abs"]
                and all(close(u[k], v[k]) for k in ("sse", "energy", "signed_sum")),
                "plane metric disagreement",
            )
    return dict(
        format="CFDK-detached-grid-agreement-v1",
        qualification_sha256=pin,
        pack_pins=[npin, wpin],
        regions=[
            {k: v for k, v in r.items() if k != "planes"} | {"verified_planes": len(r["planes"])}
            for r in out
        ],
        scope="checks paired voxel values, table and per-plane metrics; does not rerun ASTRA or establish physical accuracy",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qualification", type=Path)
    p.add_argument("pin")
    p.add_argument("native", type=Path)
    p.add_argument("witness", type=Path)
    a = p.parse_args()
    print(
        json.dumps(check(a.qualification, a.pin, a.native, a.witness), indent=2, allow_nan=False)
    )
