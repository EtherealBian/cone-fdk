"""Spark measurement transactions on a trusted local filesystem; no locks on source names."""

import json
import os
import stat
from contextlib import ExitStack
from pathlib import Path

from edge_job_inputs import Leases, TOOLS, digest, need
from sealed_inputs import publish_new, syncdir
from spark_publication import fresh, staging

MIB = 1 << 20


class DuplicateExposure(ValueError):
    pass


def pinned_reader(path):
    raise NotImplementedError(
        "original-path write/rename denial is not a Spark API; use the sealed job entry point"
    )


def commit_tree(out, bound, dest):
    """Sync files and nested directories before the no-replace directory rename."""
    for root, dirs, files in os.walk(out, topdown=False):
        root = Path(root)
        for name in dirs:
            need(stat.S_ISDIR((root / name).lstat().st_mode), "non-directory in staged tree")
        for name in files:
            fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            try:
                need(stat.S_ISREG(os.fstat(fd).st_mode), "non-file in staged tree")
                os.fsync(fd)
            finally:
                os.close(fd)
        syncdir(root)
    need(os.path.samestat(bound.parent.stat(), dest.parent.stat()), "output parent changed")
    publish_new(out, bound)


def run_job(kind, core, raw, job, products, opts, expected=None):
    """Raw is a caller-independent map of fixed logical input names to original paths."""
    layouts = {
        "axis": {"grid", "scan.f32", "scan.geom"},
        "pack": {"grid", "atlas"},
        "remount": {
            "reference.geom",
            "reference.f32",
            "inspected.geom",
            "inspected.f32",
            "fixture.json",
            "material.grid",
        },
    }
    need(kind in layouts, "unknown measurement job")
    allowed = layouts[kind]
    need(
        set(raw) == allowed
        or (kind == "remount" and set(raw) == allowed | {"body-witness.json"}),
        "unexpected input names",
    )
    dest = Path(job).absolute()
    if os.path.lexists(dest):
        raise FileExistsError(dest)
    fresh(dest)  # require an existing, real parent, not a symlink
    limits = {
        "grid": 4096,
        "atlas": 256 * MIB,
        "fixture.json": MIB,
        "material.grid": 4096,
        "body-witness.json": MIB,
    }
    tools = (
        ("wall-pack.exe", "kernels.ptx") if kind == "pack" else ("cone-fdk.exe", "kernels.ptx")
    )
    if expected is not None:
        need(set(expected) == set(raw), "external pins must cover every input")
    with staging(dest, ".measurement-") as tmp, ExitStack() as held:
        private = Path(tmp).resolve(strict=True)
        out = private / "result"
        build = private / "products"
        view = private / "inputs"
        view.mkdir()
        build.mkdir()
        lease = Leases(private, held)
        inputs = {}
        pins = {}
        for name, src in raw.items():
            cap = limits.get(name, 4 * 1024 * MIB if name.endswith(".f32") else 65536)
            src = Path(src)
            need(
                src.is_file() and 0 < src.stat().st_size <= cap,
                "invalid input type/size: " + name,
            )
            pins[name] = expected[name] if expected is not None else digest(src)
            inputs[name] = view / name
            lease.bind(src, inputs[name], pins[name], cap)
        for name in tools:
            lease.bind(
                Path(products) / name, build / name, TOOLS[name], 16 * MIB, name.endswith(".exe")
            )
        bound = Path(tmp).parent / dest.name
        try:
            doc = core(inputs, build, out, lease, opts)
        except DuplicateExposure:
            # This historical API consumes a job name on a duplicate exposure. Publish only a
            # refusal marker, never a partial CT result or a success manifest.
            reject = private / "refusal"
            reject.mkdir()
            (reject / "refusal.json").write_text(
                json.dumps(
                    {"reason": "duplicate projection payloads", "inputs_sha256": pins}, indent=2
                )
                + "\n"
            )
            commit_tree(reject, bound, dest)
            raise
        header = out / "manifest.json"
        need(
            not header.is_symlink() and json.loads(header.read_bytes()) == doc,
            "staged manifest mismatch",
        )
        if kind == "pack":
            outputs = {"levels.wallpack": doc["pack_sha256"]}
            if doc["materialized"]:
                outputs.update(
                    {f"level-{i:02}.wall": v for i, v in enumerate(doc["level_sha256"])}
                )
        else:
            outputs = doc["outputs_sha256"]
        expected_inputs = dict(pins)
        if kind == "axis":
            expected_inputs = {k: pins["scan" + k] for k in (".geom", ".f32")}
            need(outputs.get("coarse.grid") == pins["grid"], "coarse grid changed")
        expected_inputs.update({k: TOOLS[k] for k in tools})
        need(doc["inputs_sha256"] == expected_inputs, "staged input lineage mismatch")
        files = {}
        for path in out.rglob("*"):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            need(stat.S_ISREG(mode), "staged output is not a regular file")
            if path != header:
                files[path.relative_to(out).as_posix()] = digest(path)
        need(files == outputs, "staged output inventory/digest mismatch")
        doc["spark_inputs"] = {
            "contract": "sealed-bytes-v1",
            "pins": "supplied" if expected is not None else "local-self-consistency",
            "sources_sha256": pins,
            "adapter_sha256": digest(Path(__file__)),
        }
        header.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        commit_tree(out, bound, dest)
        return doc
