"""Linux checked snapshots, not mandatory locks on source pathnames."""
import ctypes
import fcntl
import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path

SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL


@contextmanager
def snapshot(path, expected, *, limit, exact=None, executable=False):
    """Bound memory, hash exactly the bytes copied, then expose only a sealed read fd."""
    with Path(path).open('rb', buffering=0) as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError('invalid input type or snapshot size')
        if exact is not None and info.st_size != exact:
            raise ValueError('projection length')
        flags = os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC
        if executable:
            flags |= getattr(os, 'MFD_EXEC', 0x10)  # supported by the qualified Spark kernel
        fd = os.memfd_create('cone-fdk-input', flags)
        with os.fdopen(fd, 'wb') as dst:
            left, h = info.st_size, hashlib.sha256()
            while left:
                block = source.read(min(left, 1 << 20))
                if not block:
                    raise ValueError('input shortened while snapshotting')
                dst.write(block); h.update(block); left -= len(block)
            if source.read(1) or h.hexdigest() != expected:
                raise ValueError('changed acquisition or native product')
            dst.flush(); os.fchmod(fd, 0o500 if executable else 0o400)
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
            if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS != SEALS:
                raise RuntimeError('kernel did not seal the snapshot')
            read_fd = os.open(f'/proc/self/fd/{fd}', os.O_RDONLY | os.O_CLOEXEC)
    try:
        yield read_fd
    finally:
        os.close(read_fd)


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_new(stage, dest):
    """Atomic same-filesystem directory publication. No fallback to replace()."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    syncdir(stage)
    if rename(-100, os.fsencode(stage), -100, os.fsencode(dest), 1):  # RENAME_NOREPLACE
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(dest))
    syncdir(dest.parent)
