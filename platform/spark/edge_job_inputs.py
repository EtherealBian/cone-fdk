"""Linux input leases and transactional output for path-based witnessed edge reviews.

Directory API for trusted local jobs. The parent manifest digest can authenticate a
source; a manifest discovered from a local directory only checks self-consistency.
The CUDA products, including both ELF entry points, have independent qualified pins.
"""

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

from sealed_inputs import snapshot
from spark_publication import fresh, publish, staging

MIB = 1 << 20
TOOLS = {
    "cone-fdk.exe": "144a9d873a6f3047637303dff2cdfe4dbe2cb1890b1d8945b1439b5d5da60883",
    "wall-pack.exe": "6351b4db3aa52e1df6a9651ccc329dcb238c45c6cdbc498dbd0c6cef9a72e7dc",
    "kernels.ptx": "2ff008d243a421c97c93f6c94031484d9ec165084855173386e6a491dce2d167",
}
CAP = {
    "body-witness.json": MIB,
    "material.grid": 4096,
    "material-loss.json": 32 * MIB,
    **{
        r + "/" + n: cap
        for r in ("reference", "inspected")
        for n, cap in (
            ("scanner.grid", 4096),
            ("shells.f32", 256 * MIB),
            ("material.wall", 32 * MIB),
        )
    },
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pin(value):
    need(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "SHA-256 required")
    return value


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Leases:
    def __init__(self, scratch, held):
        self.scratch = scratch
        self.held = held
        self.inputs = {}
        self.live_generated = 0

    @staticmethod
    def alias(fd):
        return Path(f"/proc/{os.getpid()}/fd/{fd}")

    def bind(self, source, dest, expected, cap, executable=False):
        fd = self.held.enter_context(
            snapshot(source, pin(expected), limit=cap, executable=executable)
        )
        dest.parent.mkdir(exist_ok=True)
        target = self.alias(fd)
        dest.symlink_to(target)
        self.inputs[dest] = target

    @contextmanager
    def guard(self, path):
        path = Path(path)
        if path in self.inputs:
            target = self.inputs[path]
            need(path.is_symlink() and path.readlink() == target, "input alias changed")
            yield
            need(path.is_symlink() and path.readlink() == target, "input alias changed")
            return
        # Only the freshly produced, private files may be temporarily replaced by
        # a sealed view. Never rename, lock, or change an original acquisition.
        need(
            path.is_relative_to(self.scratch)
            and path == path.resolve(strict=True)
            and stat.S_ISREG(path.lstat().st_mode),
            "unexpected generated path",
        )
        size = path.stat().st_size
        need(
            0 < size <= 256 * MIB and self.live_generated + size <= 512 * MIB,
            "generated snapshot limit",
        )
        with snapshot(path, digest(path), limit=size, exact=size) as fd:
            target = self.alias(fd)
            with tempfile.TemporaryDirectory(dir=path.parent, prefix=".edge-lease-") as tmp:
                alias = Path(tmp) / "alias"
                alias.symlink_to(target)
                os.replace(alias, path)
                self.live_generated += size
                try:
                    yield
                    need(
                        path.is_symlink() and path.readlink() == target, "generated alias changed"
                    )
                    restored = Path(tmp) / "regular"
                    with target.open("rb") as src, restored.open("xb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    os.replace(restored, path)
                finally:
                    self.live_generated -= size


def review(
    core,
    parent_job,
    job,
    levels,
    *,
    products,
    parent_sha256,
    budget,
    source_products=None,
    packed=False,
    companion_sha256=None,
):
    parent = Path(parent_job).resolve(strict=True)
    dest = Path(job).absolute()
    need(not dest.resolve(strict=False).is_relative_to(parent), "write outside parent")
    if os.path.lexists(dest):
        raise FileExistsError(dest)
    fresh(dest)
    need(type(budget) is int and 1 <= budget <= 65536, "invalid native budget")
    if packed:
        pin(parent_sha256)
        need(pin(companion_sha256) == TOOLS["wall-pack.exe"], "incorrect companion digest")
    mp = parent / "manifest.json"
    need(0 < mp.stat().st_size <= 4 * MIB, "parent manifest size")
    external = parent_sha256 is not None
    if source_products is not None:
        need(external, "archived producer requires externally pinned parent")
        source_products = Path(source_products).resolve(strict=True)
        need(
            source_products.is_dir()
            and not dest.resolve(strict=False).is_relative_to(source_products),
            "write outside source products",
        )
    pp = pin(parent_sha256) if external else digest(mp)
    with staging(dest, ".sealed-edge-") as tmp, ExitStack() as stack:
        anchored_tmp = Path(tmp)
        scratch = anchored_tmp.resolve(strict=True)
        view = scratch / "parent"
        view.mkdir()
        build = scratch / "products"
        build.mkdir()
        lease = Leases(scratch, stack)
        lease.bind(mp, view / "manifest.json", pp, 4 * MIB)
        manifest = json.loads((view / "manifest.json").read_bytes())
        need(
            isinstance(manifest, dict)
            and manifest.get("format") == "CFDK-remount-witness-job-v1"
            and manifest.get("status") in ("compared", "incomplete_coverage"),
            "unsupported parent",
        )
        expected = manifest.get("outputs_sha256")
        sources = manifest.get("inputs_sha256")
        need(isinstance(expected, dict) and isinstance(sources, dict), "missing source table")
        if source_products is None:
            provenance = build
            for key in ("cone-fdk.exe", "kernels.ptx"):
                need(sources.get(key) == TOOLS[key], "unqualified parent tool: " + key)
        else:
            provenance = scratch / "source-products"
            provenance.mkdir()
            # These authenticated producer bytes are data, never an execution path.
            for key in ("cone-fdk.exe", "kernels.ptx"):
                lease.bind(source_products / key, provenance / key, sources.get(key), 16 * MIB)

        for key, cap in CAP.items():
            lease.bind(parent / key, view / key, expected.get(key), cap)
        for key in ("cone-fdk.exe", "kernels.ptx") + (("wall-pack.exe",) if packed else ()):
            lease.bind(
                Path(products) / key, build / key, TOOLS[key], 16 * MIB, key.endswith(".exe")
            )
        out = scratch / "review"
        args = dict(
            budget=budget,
            parent_sha256=pp,
            products=build,
            source_products=provenance,
            pinned_reader=lease.guard,
        )
        if packed:
            args["companion_sha256"] = companion_sha256
        doc = core(view, out, levels, **args)
        header = out / "manifest.json"
        need(json.loads(header.read_bytes()) == doc, "staged manifest differs")
        outputs = doc["outputs_sha256"]
        need(
            set(p.name for p in out.iterdir()) == set(outputs) | {"manifest.json"},
            "unexpected staged member",
        )
        for name, h in outputs.items():
            p = out / name
            need(
                p.parent == out
                and p == p.resolve(strict=True)
                and stat.S_ISREG(p.lstat().st_mode)
                and digest(p) == h,
                "staged bytes differ",
            )
        doc["input_lease"] = {
            "format": "CFDK-GB10-directory-edge-v1",
            "parent_anchor": "supplied" if external else "self-consistency-only",
            "adapter_sha256": digest(Path(__file__)),
            "parent_products_sha256": {k: sources[k] for k in ("cone-fdk.exe", "kernels.ptx")},
            "separate_source_products": source_products is not None,
            "executables_sha256": {k: TOOLS[k] for k in TOOLS if k != "wall-pack.exe" or packed},
        }
        header.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        publish(anchored_tmp / "review", dest)
        return doc
