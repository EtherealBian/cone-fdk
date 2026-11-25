"""Implementation for the generated Linux slab_replay module. Use build_slab_compat first."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from _spark_slab_protocol import paths, run_native


def generate(plan_file, pp, scan, build, cache, dest, *, native=run_native, committed=None):
    """Run the legacy API with sealed input bytes (not locks on original pathnames).

    Callbacks are trusted synchronous same-user code. Their input names remain valid
    for this callback only, including in child processes on the qualified system.
    The default path uses inherited descriptors directly and needs no symlink view.
    """
    import _spark_slab_core as core

    plan_file, scan, build, cache, dest = map(Path, (plan_file, scan, build, cache, dest))
    if not cache.parent.is_dir() or not dest.parent.is_dir():
        raise ValueError("fresh separate outputs require existing private parents")
    if native is run_native:
        return core.generate(plan_file, pp, scan, build, cache, dest, committed=committed)
    if not callable(native):
        raise TypeError("native callback must be callable")

    def legacy_worker(fds, raw, plan, slab):
        with TemporaryDirectory(prefix="inputs-", dir=raw.parent) as tmp:
            view = Path(tmp)
            scan_alias, build_alias = view / "scan", view / "products"
            build_alias.mkdir(mode=0o700)
            for name, fd in zip(paths(scan_alias, build_alias), fds, strict=True):
                # /proc/self in a spawned legacy worker would refer to the wrong process.
                name.symlink_to(f"/proc/{os.getpid()}/fd/{fd}")
            return native(scan_alias, build_alias, raw, plan, slab)

    return core.generate(
        plan_file, pp, scan, build, cache, dest, runner=legacy_worker, committed=committed
    )
