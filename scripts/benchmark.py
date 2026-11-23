"""Cold-process benchmark on Windows/WDDM. WDDM process-local bytes sampled at ~10 ms;
not an allocation trace. CUDA counters are logged separately by the native CLI.
The two programs use their own runtime startup and output writers. This measures CLI
latency, not a claim about raw algorithmic speed. No competing benchmark is launched.
"""

import ctypes as C
import json
import os
import pathlib
import subprocess
import sys
import time

from common import BUILD, DATA, REPORT, dll_dirs

U = C.c_uint32
Q = C.c_uint64
P = C.c_void_p


class Adapter(C.Structure):
    _fields_ = [("handle", U), ("low", U), ("high", U), ("sources", U), ("precise", U)]


class Enum(C.Structure):
    _fields_ = [("count", U), ("table", C.POINTER(Adapter))]


class Query(C.Structure):
    _fields_ = [
        ("process", P),
        ("adapter", U),
        ("group", U),
        ("budget", Q),
        ("used", Q),
        ("reserved", Q),
        ("available", Q),
        ("physical", U),
    ]


class Sampler:
    def __init__(self):
        self.dll = C.WinDLL("gdi32")
        arr = (Adapter * 32)()
        req = Enum(32, arr)
        status = self.dll.D3DKMTEnumAdapters2(C.byref(req))
        if status:
            raise OSError(f"D3DKMTEnumAdapters2: {status}")
        self.adapters = [int(a.handle) for a in arr[: req.count]]

    def sample(self, proc):
        total = 0
        ok = False
        for a in self.adapters:
            q = Query(int(proc._handle), a, 0)
            status = self.dll.D3DKMTQueryVideoMemoryInfo(C.byref(q))
            if not status:
                total += q.used
                ok = True
        return total if ok else None

    def close(self):
        for a in self.adapters:
            self.dll.D3DKMTCloseAdapter(C.byref(U(a)))


def launch(cmd, env, monitor):
    t = time.perf_counter()
    peak = 0
    count = 0
    with subprocess.Popen(
        list(map(str, cmd)),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as p:
        while p.poll() is None:
            m = monitor.sample(p)
            if m is not None:
                peak = max(peak, m)
                count += 1
            time.sleep(0.01)
        out, err = p.communicate()
        elapsed = time.perf_counter() - t
        if p.returncode:
            raise RuntimeError(err)
    return dict(
        elapsed_seconds=elapsed,
        wddm_local_peak_bytes=peak,
        wddm_samples=count,
        metrics=json.loads(out),
    )


def run():
    monitor = Sampler()
    env = os.environ.copy()
    # Windows venv redirects python.exe to a child; sample the actual GPU owner.
    env["__PYVENV_LAUNCHER__"] = sys.executable
    env["PATH"] = os.pathsep.join(map(str, dll_dirs())) + os.pathsep + env["PATH"]
    scan = DATA / "phantom"
    res = {}
    try:
        for n in (256, 512):
            out = DATA / "bench-native.f32"
            native = [
                BUILD / "cone-fdk.exe",
                "--ptx",
                BUILD / "kernels.ptx",
                "--geom",
                scan.with_suffix(".geom"),
                "--input",
                scan.with_suffix(".f32"),
                "--output",
                out,
                "--n",
                n,
                "--fov",
                32,
                "--budget",
                768,
                "--batch",
                16,
            ]
            ref = [
                sys._base_executable,
                pathlib.Path(__file__).parent / "reconstruct.py",
                scan,
                DATA / "bench-astra.f32",
                "--n",
                n,
                "--fov",
                32,
                "--astra",
            ]
            d = {"native": [], "astra": []}
            for i in range(3):
                for key, cmd in (
                    [("native", native), ("astra", ref)]
                    if i % 2 == 0
                    else [("astra", ref), ("native", native)]
                ):
                    d[key].append(launch(cmd, env, monitor))
            res[str(n)] = d
            print(
                n,
                [
                    (
                        k,
                        [(round(x["elapsed_seconds"], 3), x["wddm_local_peak_bytes"]) for x in v],
                    )
                    for k, v in d.items()
                ],
                flush=True,
            )
    finally:
        monitor.close()
    (REPORT / "cold-benchmark.json").write_text(json.dumps(res, indent=2))
    assert all(
        x["wddm_local_peak_bytes"] > 0
        for config in res.values()
        for runs in config.values()
        for x in runs
    ), "WDDM residency telemetry unavailable"


if __name__ == "__main__":
    run()
