"""Derive a separate dense/single-ROI GB10 FDK product. Never patch the frozen checkout build."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from build_residual import HERE, ROOT, sha
from spark_publication import fresh, staging, publish

PARENT = "a8fa63ad1ac11ecf7825a7b168f321891f2afd676223f991839e138724b3f432"
BASE = "144a9d873a6f3047637303dff2cdfe4dbe2cb1890b1d8945b1439b5d5da60883"
PTX = "2ff008d243a421c97c93f6c94031484d9ec165084855173386e6a491dce2d167"
EXTRAS = ("finite_sve.hpp", "coherent.hpp", "attributes.cpp")


def derive(s):
    def edit(old, new):
        nonlocal s
        if s.count(old) != 1:
            raise ValueError("unknown upstream FDK boundary: " + old)
        s = s.replace(old, new)

    edit("#include <utility>", '#include <utility>\n#include "coherent.hpp"')
    start = s.index("        if (argc >= 2")
    end = s.index("        auto begin", start)
    s = (
        s[:start]
        + """        require_sve();
        if (argc == 2 && argv[1] == "--help") {
            std::cout << "GB10 dense/single-ROI FDK: --gb10-transport staged|shared plus original required geometry, input, output, n, fov, budget, batch and PTX options.\\n";
            return 0;
        }
"""
        + s[end:]
    )
    edit(
        '"--fov",  "--budget", "--batch",  "--ptx"',
        '"--fov",  "--budget", "--batch",  "--ptx", "--gb10-transport"',
    )
    edit(
        "        auto mode =",
        """        auto flavour = options.at("--gb10-transport");
        bool shared=flavour=="shared";
        if ((!shared && flavour!="staged") || options.count("--roi-list") || options.count("--plane") || options.count("--stack") || options.count("--cylinder") || options.count("--host-memory"))
            throw std::runtime_error("expected qualified dense/ROI GB10 mode");
        auto mode =""",
    )
    edit(
        "        size_t free0, total;",
        "        spark_require_ats(gpu);\n        size_t free0, total;",
    )
    edit(
        "Buf block(gpu, elems * 4), views(gpu, row * batch * 4),\n            freq",
        "SharedBuf block(gpu,elems*4,shared), views(gpu,row*batch*4,shared);\n        Buf freq",
    )
    edit(
        'HostBuffer host(gpu, row * batch, mode == "auto"),\n            outChunk(gpu, std::min<size_t>(elems, 2u << 20), mode == "auto");',
        """size_t chunk = std::min<size_t>(elems, 2u << 20);
        std::unique_ptr<HostBuffer> host,outChunk;
        if(!shared){host=std::make_unique<HostBuffer>(gpu,row*batch,true);outChunk=std::make_unique<HostBuffer>(gpu,chunk,true);}
        size_t h2dBulk=0,d2hBulk=0;""",
    )
    edit(
        "in.transfer(host.data(), size, false);",
        "float* input = shared ? views.cpu() : host->data();\n                in.transfer(input,size,false);",
    )
    edit("!finite_block(host.data(),", "!finite_block(input,")
    edit(
        "timer.start();\n                CK(gpu.cuMemcpyHtoD_v2(views.p, host.data(), size));\n                h2d += timer.stop();",
        "if(!shared){timer.start();CK(gpu.cuMemcpyHtoD_v2(views.p,input,size));h2d+=timer.stop();h2dBulk+=size;}",
    )
    edit("off += outChunk.count", "off += chunk")
    edit("std::min(outChunk.count,", "std::min(chunk,")
    edit(
        "timer.start();\n                CK(gpu.cuMemcpyDtoH_v2(outChunk.data(), block.p + 4 * off, bytesOut));\n                d2h += timer.stop();",
        "float* ready = shared ? block.cpu()+off : outChunk->data();\n                if(!shared){timer.start();CK(gpu.cuMemcpyDtoH_v2(ready,block.p+4*off,bytesOut));d2h+=timer.stop();d2hBulk+=bytesOut;}",
    )
    edit("!finite_block(outChunk.data(),", "!finite_block(ready,")
    edit("output.file->transfer(outChunk.data(),", "output.file->transfer(ready,")
    edit(
        "for (Buf *b : {&block, &views, &freq, &cs, &h, &work})",
        "CK(gpu.cuCtxSynchronize());block.verify();views.verify();\n        for (Buf *b : {&freq, &cs, &h, &work})",
    )
    edit(
        "<< host.pinnedBytes() + outChunk.pinnedBytes()",
        """<< (shared?0:host->pinnedBytes()+outChunk->pinnedBytes())
                  << ",\\"gb10_transport\\":\\"" << flavour << "\\",\\"sve_bits\\":" << spark_vector_bits()
                  << ",\\"bulk_h2d_bytes\\":" << h2dBulk << ",\\"bulk_d2h_bytes\\":" << d2hBulk
                  << ",\\"shared_alloc_bytes\\":" << (shared?block.sz+views.sz+4*Buf::pad:0)""",
    )
    edit(
        "        const int requestedBatch",
        '        if(shared && budget>(768ull<<20)) throw std::runtime_error("shared budget exceeds qualified ceiling");\n        const int requestedBatch',
    )
    return s


def build(out):
    out = fresh(out)
    baseline = ROOT / "build/spark"
    upstream = baseline / "src"
    src = HERE / "fdk"
    assert platform.machine() == "aarch64" and platform.system() == "Linux"

    def pins():
        files = sorted(p for p in upstream.iterdir() if p.suffix in (".cpp", ".hpp"))
        d = {p.name: sha(p) for p in files}
        key = hashlib.sha256("".join(f"{k} {v}\n" for k, v in d.items()).encode()).hexdigest()
        assert (
            key == PARENT
            and sha(baseline / "cone-fdk") == BASE
            and sha(baseline / "kernels.ptx") == PTX
        )
        return d, {k: sha(src / k) for k in EXTRAS}

    orig, patch = pins()
    hashes = []
    with staging(out, ".coherent-build-") as t:
        stage = Path(t) / "stage"
        stage.mkdir()
        for idx in range(2):
            with TemporaryDirectory(dir=Path(t).resolve()) as td:
                wd = Path(td)
                [shutil.copyfile(upstream / k, wd / k) for k in orig]
                assert all(sha(wd / k) == v for k, v in orig.items())
                [shutil.copyfile(src / k, wd / k) for k in EXTRAS]
                assert all(sha(wd / k) == v for k, v in patch.items())
                shutil.copyfile(wd / "finite_sve.hpp", wd / "finite.hpp")
                main = wd / "main.cpp"
                main.write_text(derive(main.read_text()))
                generated = sha(main)
                subprocess.run(
                    [
                        "g++",
                        "-std=c++14",
                        "-O3",
                        "-I/usr/local/cuda/include",
                        "main.cpp",
                        "attributes.cpp",
                        "-ldl",
                        "-o",
                        "coherent-fdk",
                    ],
                    cwd=wd,
                    check=True,
                )
                elf = wd / "coherent-fdk"
                hashes.append(sha(elf))
                if not idx:
                    shutil.copy2(elf, stage / elf.name)
        assert hashes[0] == hashes[1] == sha(stage / "coherent-fdk") and pins() == (orig, patch)
        doc = dict(
            format="CFDK-GB10-coherent-FDK-build-v1",
            source_parent=PARENT,
            original=BASE,
            ptx=PTX,
            patch=patch,
            builder_sha256=sha(__file__),
            generated_main_sha256=generated,
            independent_elf_sha256=hashes,
            compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        )
        (stage / "build.json").write_text(json.dumps(doc, indent=2) + "\n")
        publish(stage, out)
    print(sha(out / "build.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    build(p.parse_args().out)
