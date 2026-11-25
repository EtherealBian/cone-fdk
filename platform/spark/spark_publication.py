"""Local Linux publication: no destination symlinks or replacement, hold the parent fd.

Like the native qualification, this assumes a trusted same-user checkout. A parent renamed
mid-run cannot redirect publication to a different directory. No network-FS guarantee.
"""
import os
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from sealed_inputs import publish_new, syncdir


@contextmanager
def anchored(target):
    path = Path(target)
    if sys.platform != 'linux' or '..' in path.parts or not path.name:
        raise ValueError('Linux destination with no parent traversal required')
    path = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open('/', flags)
    try:
        for component in path.parent.parts[1:]:
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        try:
            os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError('destination must be fresh, including dangling links')
        # Child CUDA processes reopen paths; /proc/self/fd would name a different process.
        yield Path(f'/proc/{os.getpid()}/fd/{fd}') / path.name
    finally:
        os.close(fd)


def fresh(target):
    with anchored(target):
        return Path(target).absolute()


@contextmanager
def staging(target, prefix):
    with anchored(target) as bound:
        with tempfile.TemporaryDirectory(dir=bound.parent, prefix=prefix) as tmp:
            yield tmp


def publish(stage, target):
    # stage is one directory below TemporaryDirectory in the held destination parent.
    bound = stage.parent.parent / target.name
    if not os.path.samestat(bound.parent.stat(), target.parent.stat()):
        raise ValueError('destination parent changed since staging')
    for entry in stage.iterdir():
        fd = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError('unexpected staged non-file')
            os.fsync(fd)
        finally:
            os.close(fd)
    publish_new(stage, bound)


def atomic_index(writer, target, data):
    with anchored(target) as bound:
        writer(bound, data)  # original hard-link publisher already fsyncs payload
        syncdir(bound.parent)
