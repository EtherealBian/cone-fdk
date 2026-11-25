"""Build a separate tile-search GB10 executable and PTX without changing any math or source baselines."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from build_coherent_fdk import BASE, PARENT, PTX, ROOT, HERE, sha
from residual_witness import need
from spark_publication import fresh, publish, staging

KERNEL = "a61ff35da7b739c65badd31c9364aa7b441aa937ff6a2189000e7c0c456d7cba"
GEOM = "a3af5151ca0774118f3671dc2e1e2fa98d07684a72365b998fbf45e4348da877"
TILES = (
    (16, 8, 16),
    (32, 4, 8),
    (32, 4, 16),
    (32, 4, 32),
    (16, 8, 8),
    (16, 8, 32),
    (8, 16, 16),
    (32, 8, 16),
    (8, 8, 16),
)


def once(s, a, b):
    need(s.count(a) == 1, "unrecognized frozen source boundary")
    return s.replace(a, b)


def kernel(s):
    s = once(s, "template <bool roi>", "template <bool roi, int lanes>")
    s = once(
        s,
        "    constexpr int lanes = 16;",
        '    static_assert(lanes % 4 == 0, "preserve original rounding groups");',
    )
    a = s.index('extern "C" __global__ void back(')
    e = s.index("// Shared point sampler", a)
    wrappers = s[a:e]
    kernels = []
    for z in (8, 16, 32):
        w = once(
            once(wrappers, "back_tile<false>", f"back_tile<false,{z}>"),
            "back_tile<true>",
            f"back_tile<true,{z}>",
        )
        suffix = "" if z == 16 else str(z)
        w = once(
            once(w, "void back(", f"void back{suffix}("),
            "void back_roi(",
            f"void back_roi{suffix}(",
        )
        kernels.append(w)
    return s[:a] + "".join(kernels) + s[e:]


def host(s):
    start = s.index("        if (argc >= 2")
    end = s.index("        auto begin", start)
    s = s[:start] + s[end:]
    s = once(s, '"--batch",  "--ptx"', '"--batch",  "--ptx", "--gb10-tile"')
    opts = ", ".join('"' + "x".join(map(str, t)) + '"' for t in TILES)
    tups = ", ".join("{" + ",".join(map(str, t)) + "}" for t in TILES)
    s = once(
        s,
        "        auto mode =",
        f"""        const std::vector<std::string> tileNames={{{opts}}};
        const int tileShape[][3]={{{tups}}};
        auto choice=std::find(tileNames.begin(),tileNames.end(),options.at("--gb10-tile"));
        if(choice==tileNames.end() || options.count("--roi-list") || options.count("--plane") || options.count("--stack") || options.count("--cylinder"))throw std::runtime_error("supported dense/single ROI tile required");
        const int* tile=tileShape[choice-tileNames.begin()];
        int bx=tile[0],by=tile[1],bz=tile[2];
        auto mode =""",
    )
    s = once(
        s,
        "        size_t free0, total;",
        """        if(std::string(gpu.name)!="NVIDIA GB10") throw std::runtime_error("GB10 required");
        std::string entry=std::string(cropped?"back_roi":"back")+(bz==16?std::string():std::to_string(bz));
        size_t free0, total;""",
    )
    start = s.index("             back = gpu.kernel(cylindrical")
    end = s.index("             un =", start)
    s = s[:start] + "             back = gpu.kernel(entry.c_str()),\n" + s[end:]
    s = once(
        s,
        "up(b.nx, 16), up(b.ny, 8), up(b.nz + (b.z & 3), 16), 16, 8,",
        "up(b.nx, bx), up(b.ny, by), up(b.nz + (b.z & 3), bz), bx, by,",
    )
    s = once(
        s,
        "up(g.n, 16), up(g.n, 8), up(b.nz, 16), 16, 8,",
        "up(g.n, bx), up(g.n, by), up(b.nz, bz), bx, by,",
    )
    s = once(
        s,
        '<< ",\\"bp_ms\\":" << bp',
        '<< ",\\"gb10_tile\\":[" << bx << "," << by << "," << bz << "]" << ",\\"bp_ms\\":" << bp',
    )
    return s


def build(out):
    need(platform.machine() == "aarch64" and platform.system() == "Linux", "native Spark only")
    out = fresh(out)
    base = ROOT / "build/spark"
    src = base / "src"
    d = {p.name: sha(p) for p in sorted(src.iterdir()) if p.suffix in (".hpp", ".cpp")}
    need(
        hashlib.sha256("".join(f"{k} {v}\n" for k, v in d.items()).encode()).hexdigest() == PARENT
        and d["geometry.hpp"] == GEOM
        and sha(base / "cone-fdk") == BASE
        and sha(base / "kernels.ptx") == PTX,
        "parent build",
    )
    seen = []
    with staging(out, ".tile-build-") as t:
        stage = Path(t) / "stage"
        stage.mkdir()
        for i in range(2):
            with TemporaryDirectory(dir=Path(t).resolve()) as tmp:
                wd = Path(tmp)
                for n, h in dict(d, **{"kernels.cu": KERNEL}).items():
                    shutil.copyfile(src / n, wd / n)
                    need(sha(wd / n) == h, "source changed")
                for n, f in [("main.cpp", host), ("kernels.cu", kernel)]:
                    p = wd / n
                    p.write_text(f(p.read_text()))
                subprocess.run(
                    ["g++", "-O3", "-std=c++14", "main.cpp", "-ldl", "-o", "tile-fdk"],
                    cwd=wd,
                    check=True,
                )
                subprocess.run(
                    [
                        "/usr/local/cuda/bin/nvcc",
                        "-ptx",
                        "-arch=compute_121",
                        "--std=c++17",
                        "kernels.cu",
                        "-o",
                        "tile.ptx",
                    ],
                    cwd=wd,
                    check=True,
                )
                seen.append(
                    {n: sha(wd / n) for n in ("main.cpp", "kernels.cu", "tile-fdk", "tile.ptx")}
                )
                if not i:
                    for n in ("tile-fdk", "tile.ptx"):
                        shutil.copy2(wd / n, stage / n)
        need(
            seen[0] == seen[1]
            and all(sha(stage / n) == seen[0][n] for n in ("tile-fdk", "tile.ptx")),
            "independent build mismatch",
        )
        rec = dict(
            format="CFDK-GB10-tile-build-v1",
            builder_sha256=sha(__file__),
            parent_sources=PARENT,
            kernel_parent=KERNEL,
            geometry=GEOM,
            original=BASE,
            original_ptx=PTX,
            tile_xyz=TILES,
            independent_products=seen,
        )
        (stage / "build.json").write_text(json.dumps(rec, indent=2) + "\n")
        publish(stage, out)
    print(sha(out / "build.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
