"""Build the GB10 cross-grid host controller twice; all filtering and backprojection use the pinned original PTX."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from build_tile_fdk import BASE, PTX, ROOT, HERE, PARENT, KERNEL, once, sha, need
from spark_publication import fresh, staging, publish


def derive(s):
    s = once(s, "#include <utility>", '#include <utility>\n#include "multigrid.hpp"')
    s = s[: s.index("        if (argc >= 2")] + s[s.index("        auto begin") :]
    a = s.index("        std::set<std::string> keys")
    e = s.index("        for (const auto &key : keys)", a)
    s = (
        s[:a]
        + """        std::set<std::string> keys = {"--geom","--input","--output","--ptx","--batch","--budget","--grids"};
        for(int i=1;i<argc;i+=2){if(i+1>=argc || !keys.count(argv[i]) || options.count(argv[i]))throw std::runtime_error("unknown/duplicate/incomplete argument");options[argv[i]]=argv[i+1];}
        const std::string mode="auto",policy="batch-first"; const bool packed=true,sliced=false,cylindrical=false,nrrd=false,stacked=false;
"""
        + s[e:]
    )
    a = s.index("        if (packed && same_file")
    e = s.index("        Geom g", a)
    s = (
        s[:a]
        + """        if(same_file(dest,absolute(val("--grids"))))throw std::runtime_error("output aliases grid list");
"""
        + s[e:]
    )
    a = s.index("        g.n = integer")
    e = s.index("        if (!(g.a", a)
    s = (
        s[:a]
        + """        const auto requests=read_grid_requests(val("--grids"),g); g.n=requests[0].n;g.vox=requests[0].pitch;const float fov=requests[0].fov;
        const bool cropped=true;std::vector<Box> boxes;for(auto &r:requests)boxes.push_back(r.b);
"""
        + s[e:]
    )
    a = s.index("        if (cylindrical)")
    e = s.index("        size_t budget", a)
    s = s[:a] + s[e:]
    s = once(
        s,
        "        size_t free0, total;",
        """        if(std::string(gpu.name)!="NVIDIA GB10")throw std::runtime_error("GB10 required");
        size_t free0, total;""",
    )
    s = once(
        s,
        "        Buf block(gpu, elems * 4)",
        """        if(bestPasses!=1 || batch!=std::min(requestedBatch,g.a))throw std::runtime_error("one pass at requested batch cannot fit: reduce output ROIs or raise the buffer budget");
        Buf block(gpu, elems * 4)""",
    )
    s = once(
        s,
        "write_pack_header(*output.file, boxes, g.n, g.vox);",
        "write_multigrid(*output.file,requests);",
    )
    a = s.index("                for (const auto &b : tiles)")
    e = s.index("                bp += timer.stop();", a)
    s = (
        s[:a]
        + """                if(tiles.size()!=requests.size())throw std::runtime_error("unexpected tile split");
                for(size_t i=0;i<requests.size();++i){
                    const auto &r=requests[i];const auto &b=r.b;const auto &t=tiles[i];
                    if(t.x!=b.x || t.y!=b.y || t.z!=b.z || t.nx!=b.nx || t.ny!=b.ny || t.nz!=b.nz)throw std::runtime_error("wrong tile order");
                    Geom gridGeom=g;gridGeom.n=r.n;gridGeom.vox=r.pitch;
                    gpu.launch(back,up(b.nx,16),up(b.ny,8),up(b.nz+(b.z&3),16),16,8,dst,freq.p,cs.p,gridGeom,first,actual,b.z,b.nz,b.x,b.y,b.nx,b.ny);
                    dst+=4*b.voxels();++launches;
                }
"""
        + s[e:]
    )
    s = once(s, "written = pack_header_bytes(boxes.size());", "written = 12+36*requests.size();")
    s = once(s, r"\"format\":\"CFDKROI1\"", r"\"format\":\"CFDKMG01\"")
    s = once(s, r"\"n\":", r"\"first_grid_n\":")
    s = once(s, r"\"voxel_spacing_mm\":", r"\"first_voxel_spacing_mm\":")
    s = once(
        s,
        r'"{\"origin_xyz\":["',
        r'"{\"n\":" << requests[i].n << ",\"fov_mm\":" << std::defaultfloat << std::setprecision(9) << requests[i].fov << ",\"pitch_mm\":" << requests[i].pitch << std::fixed << std::setprecision(4) << ",\"origin_xyz\":["',
    )
    return s


def build(out):
    need(platform.machine() == "aarch64" and platform.system() == "Linux", "Spark required")
    out = fresh(out)
    up = ROOT / "build/spark/src"
    head = HERE / "fdk/multigrid.hpp"
    d = {p.name: sha(p) for p in sorted(up.iterdir()) if p.suffix in (".hpp", ".cpp")}
    need(
        hashlib.sha256("".join(f"{n} {h}\n" for n, h in d.items()).encode()).hexdigest() == PARENT
        and sha(up / "kernels.cu") == KERNEL
        and sha(up.parent / "kernels.ptx") == PTX
        and sha(up.parent / "cone-fdk") == BASE,
        "changed parent",
    )
    h = sha(head)
    seen = []
    with staging(out, ".grid-build-") as tmp:
        stage = Path(tmp) / "ready"
        stage.mkdir()
        for i in range(2):
            with tempfile.TemporaryDirectory(dir=Path(tmp).resolve()) as st:
                wd = Path(st)
                for name, hv in d.items():
                    shutil.copyfile(up / name, wd / name)
                    need(sha(wd / name) == hv, "source race")
                shutil.copyfile(head, wd / head.name)
                need(sha(wd / head.name) == h, "header changed")
                # A private scratch command is also safe to call directly: never replace an existing pack.
                files = wd / "files.hpp"
                files.write_text(
                    once(
                        files.read_text(),
                        'if(rename(tmp.c_str(),dest.c_str()))throw ioerr("cannot publish output");published=true;',
                        'if(link(tmp.c_str(),dest.c_str()))throw ioerr("cannot publish fresh multigrid output");if(unlink(tmp.c_str()))throw ioerr("cannot unlink multigrid staging name");published=true;',
                    )
                )
                main = wd / "main.cpp"
                main.write_text(derive(main.read_text()))
                subprocess.run(
                    [
                        "g++",
                        "-std=c++14",
                        "-O3",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "main.cpp",
                        "-ldl",
                        "-o",
                        "multigrid",
                    ],
                    cwd=wd,
                    check=True,
                )
                seen.append({n: sha(wd / n) for n in ["main.cpp", "files.hpp", "multigrid"]})
                if not i:
                    shutil.copy2(wd / "multigrid", stage / "multigrid")
        need(
            seen[0] == seen[1] and sha(stage / "multigrid") == seen[0]["multigrid"],
            "unreproducible build",
        )
        (stage / "build.json").write_text(
            json.dumps(
                dict(
                    format="CFDK-GB10-multigrid-build-v1",
                    builder=sha(__file__),
                    header=h,
                    parent=PARENT,
                    ptx=PTX,
                    original=BASE,
                    builds=seen,
                ),
                indent=2,
            )
            + "\n"
        )
        publish(stage, out)
    print(sha(out / "build.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
