"""Exercise pack portability and real process-death recovery on the original GB10 campaign."""

import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import offline_campaign as oc
from pack_campaign import pack, write
from resume_campaign import batch, publish

HERE = Path(__file__).resolve().parent
PIN = "fcaec11cafcbec9aae1cbab576252d5aee81b6b9b623e0a8caef56f7a0eda91c"
SCOPE = "Offline sampled-grid agreement for eight simulated comparison jobs; no raw-wall replay, physical specimens, physical defect or dimensional accuracy in this qualification."


class HashWriter:
    def __init__(self):
        self.h = hashlib.sha256()
        self.size = 0

    def tell(self):
        return self.size

    def write(self, b):
        self.h.update(b)
        self.size += len(b)
        return len(b)


def verify(source, job):
    source, job = (Path(x).absolute() for x in (source, job))
    job.mkdir(mode=0o700, exist_ok=False)
    e = job / "evidence.cfdk"
    direct = pack(source, e, PIN)
    expected = oc.encoded(direct) + b"\n"
    sink = HashWriter()
    write(sink, source, PIN)
    wire = e.stat().st_size
    oc.need(
        sink.size == wire and sink.h.hexdigest() == oc.sha(e.read_bytes()),
        "nondeterministic wire bytes",
    )
    # Run only a copy of the reviewer and a renamed pack, with no checkout in cwd/sys.path.
    with tempfile.TemporaryDirectory(dir=job) as name:
        alone = Path(name)
        tool = alone / "review.py"
        shutil.copyfile(HERE / "offline_campaign.py", tool)
        os.link(e, alone / "sealed.input")
        offline = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "review.py", "sealed.input", PIN],
            cwd=alone,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            capture_output=True,
            check=True,
        )
        oc.need(
            offline.stdout == expected and not offline.stderr, "relocated offline result differs"
        )
    receipt = job / "review.json"
    publish(receipt, expected)
    # Keep no extra copy of the same final report after verifying both outputs.
    # Production producers never overwrite; ephemeral qualification outputs live in a new dir.
    with tempfile.TemporaryDirectory(dir=job) as name:
        scratch = Path(name)
        cold = batch(e, PIN, scratch / "state", scratch / "cold.json")
        oc.need((scratch / "cold.json").read_bytes() == expected, "cold receipt differs")
        state = job / "resume-state"
        rest = scratch / "resumed.json"
        code = "import os,sys;sys.path.insert(0,sys.argv[1]);from resume_campaign import batch;batch(*sys.argv[2:],after_commit=lambda n:os._exit(73) if n==19 else None)"
        r = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                code,
                str(HERE),
                str(e),
                PIN,
                str(state),
                str(rest),
            ],
            capture_output=True,
        )
        oc.need(r.returncode == 73 and not rest.exists(), "wrong crash/publication outcome")
        db = sqlite3.connect(f"file:{state / 'jobs.sqlite3'}?mode=ro", uri=True)
        try:
            count = db.execute("select count(*) from jobs").fetchone()[0]
        finally:
            db.close()
        oc.need(count == 19, "wrong durable prefix")
        resumed = batch(e, PIN, state, rest, resume=True)
        oc.need(
            rest.read_bytes() == expected
            and resumed["reused"] == count
            and resumed["fresh"] == 48
            and cold["fresh"] == 67
            and cold["reused"] == 0,
            "recovery mismatch",
        )
    numbers = {
        k: direct[k]
        for k in ("maps", "cells", "logical_payload_bytes", "unique_frames", "refused")
    }
    chain = oc.obj((source / "chain.json").read_bytes())
    names = (
        "offline_campaign.py",
        "pack_campaign.py",
        "resume_campaign.py",
        "qualify_detached.py",
        "tests/test_detached_campaign.py",
    )
    result = dict(
        format="CFDK-detached-campaign-qualification-v1",
        scope=SCOPE,
        chain_sha256=PIN,
        files={n: oc.sha((HERE / n).read_bytes()) for n in names},
        pack_sha256=sink.h.hexdigest(),
        wire_bytes=wire,
        **numbers,
        logical_frame_occurrences=2 + 6 * len(chain["rows"]),
        deterministic_second_build=True,
        detached_python=sys.version.split()[0],
        detached_options=["-I", "-S", "-B"],
        isolated_copy_verified=True,
        hard_exit={
            "code": r.returncode,
            "committed_before_exit": count,
            "partial_publication": False,
        },
        cold=cold,
        resumed=resumed,
        review_sha256=oc.sha(expected),
        limitations=[
            "Published lineage, rather than an independently repeated scan acquisition, is the external trust root.",
            "Trust local account and local filesystem; no network filesystem, malicious cache author, power-cut, or restart-of-host qualification.",
            "Output contains component counts, typed sampled-cell faces, anchor state and withheld reasons; none is a physical perimeter.",
        ],
    )
    publish(job / "qualification.json", oc.encoded(result) + b"\n")
    return result


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("fresh_job", type=Path)
    a = p.parse_args()
    print(oc.encoded(verify(a.source, a.fresh_job)).decode())
