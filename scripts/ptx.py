"""Build CUDA C++ to portable compute_80 PTX with official NVIDIA NVRTC 12."""

import ctypes as C
import os
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]


def compile_ptx(source_file="src/kernels.cu", output_file="build/kernels.ptx"):
    lib = pathlib.Path(sys.prefix) / "Lib/site-packages/nvidia/cuda_nvrtc/bin"
    _cookie = os.add_dll_directory(str(lib))
    os.environ["PATH"] = str(lib) + os.pathsep + os.environ["PATH"]
    _builtins = C.CDLL(str(next(lib.glob("nvrtc-builtins64*.dll"))))
    dll = C.CDLL(str(next(p for p in lib.glob("nvrtc64_*.dll") if ".alt" not in p.name)))
    prog = C.c_void_p()
    source = (root / source_file).read_bytes()

    def ck(n):
        if n:
            raise RuntimeError(f"NVRTC status {n}")

    ck(dll.nvrtcCreateProgram(C.byref(prog), source, b"kernels.cu", 0, None, None))
    opts = (C.c_char_p * 2)(b"--gpu-architecture=compute_80", b"--std=c++14")
    rc = dll.nvrtcCompileProgram(prog, 2, opts)
    s = C.c_size_t()
    dll.nvrtcGetProgramLogSize(prog, C.byref(s))
    log = C.create_string_buffer(s.value)
    dll.nvrtcGetProgramLog(prog, log)
    print(log.value.decode(), end="")
    ck(rc)
    ck(dll.nvrtcGetPTXSize(prog, C.byref(s)))
    b = C.create_string_buffer(s.value)
    ck(dll.nvrtcGetPTX(prog, b))
    (root / output_file).parent.mkdir(parents=True, exist_ok=True)
    (root / output_file).write_bytes(b.raw)
    ck(dll.nvrtcDestroyProgram(C.byref(prog)))


if __name__ == "__main__":
    compile_ptx(*sys.argv[1:])
