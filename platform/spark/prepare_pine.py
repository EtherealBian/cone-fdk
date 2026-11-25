"""Prepare both measured scans on Spark only after all original files pass verification."""
import argparse
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from sealed_inputs import publish_new, snapshot

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import prepare_pine_cone as pine


@contextmanager
def sealed_original(path,pin,rec):
    # Published inputs total about 4 GiB. Never allocate from an untrusted receipt length.
    pine.need(isinstance(pin,str) and re.fullmatch('[0-9a-f]{64}',pin), 'invalid archived SHA-256')
    with snapshot(path,pin,limit=600<<20,exact=rec['bytes']) as fd:
        with os.fdopen(os.dup(fd),'rb') as f: yield f


def run(originals, destination):
    pine.need(sys.platform=='linux','Linux adapter only')
    info=destination.absolute().parent.stat()
    pine.need(info.st_uid==os.getuid() and info.st_mode & 0o077==0,'private output parent required')
    # The frozen source record size/MD5 and geometry, member CRCs, frame hashes and
    # independently reread 90-view subset are still enforced by the common core.
    return pine.prepare_verified(originals,destination,sealed_original,publish_new,
             {p.name:pine.sha(p) for p in [Path(__file__),HERE/'sealed_inputs.py']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('originals',type=Path);p.add_argument('destination',type=Path)
    run(**vars(p.parse_args()))
