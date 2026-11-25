"""Real process-death and case replay on the measured walnut; no synthetic pine results."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import physical_queue as q

BASELINE = "12f2f81c0d88d75b05fd824876b94cc236d63b5613260341c2917998a5db3349"


def run(root):
    root = Path(root).absolute()
    q.private(root.parent)
    root.mkdir(mode=0o700)
    state = root / "state"
    q.create(state)
    results = {}
    expect = []

    def save(n, d):
        p = root / (n + ".json")
        q.write(p, d)
        results[n] = q.pb.qr.sha(p)

    def cli(n, mode):
        cmd = [sys.executable, "-B", str(q.HERE / "physical_queue.py"), mode, str(state)]
        result = subprocess.run(cmd, cwd=root, check=True, capture_output=True)
        doc = json.loads(result.stdout)
        save(n, doc)
        return doc

    def crash(point):
        p = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "fault", str(state), point],
            capture_output=True,
            cwd=root,
            timeout=400,
        )
        q.need(
            p.returncode == -signal.SIGKILL and not p.stdout and not p.stderr,
            "did not die at requested boundary: " + str((p.returncode, p.stderr[:500])),
        )
        expect.append(
            dict(
                point=point,
                returncode=p.returncode,
                stdout_bytes=len(p.stdout),
                stderr_bytes=len(p.stderr),
            )
        )

    cold = cli("cold", "status")
    q.need(cold["committed_cases"] == 0 and not cold["orphan_attempts"], "cold state")
    crash("verified_before_commit")
    orphan = cli("after_uncommitted", "status")
    q.need(
        not orphan["committed_cases"] and len(orphan["orphan_attempts"]) == 1,
        "orphan was lost/promoted",
    )
    orphan_name = orphan["orphan_attempts"][0]
    old = state / "attempts" / orphan_name
    old_hash = q.pb.qr.sha(old / "result/qualification.json")
    q.need(
        q.owned_json(old / "pending.json")["report_sha256"] == old_hash,
        "no complete precommit evidence",
    )
    crash("committed")
    warm = cli("after_committed", "work")
    again = cli("again", "status")
    q.need(
        warm == again
        and warm["orphan_attempts"] == [orphan_name]
        and warm["committed_cases"] == 1
        and warm["fresh_cases"] == 0
        and [r["execution"] for r in warm["cases"]]
        == ["reused", "awaiting_external_parent", "awaiting_external_parent"],
        "resume coverage",
    )
    r = warm["cases"][0]
    q.need(
        r["attempt"] != orphan_name and len(list((state / "attempts").iterdir())) == 2,
        "did not recompute or reran a committed case",
    )
    accepted = state / "attempts" / r["attempt"] / "result"
    frozen = q.M / "residual-walnut-001"
    base = q.pb.assess("walnut720", frozen, BASELINE, None)
    orig = json.loads((frozen / "qualification.json").read_bytes())
    candidate = json.loads((accepted / "qualification.json").read_bytes())
    abandoned = json.loads((old / "result/qualification.json").read_bytes())
    q.need(
        candidate["planes"] == abandoned["planes"] == orig["planes"]
        and candidate["aggregate"] == abandoned["aggregate"] == base["aggregate"]
        and r["evidence"]["aggregate"] == base["aggregate"],
        "independent all-plane/previous qualification differs",
    )
    for n in ("candidate-slice.f32", "reference-slice.f32"):
        q.need(
            q.pb.qr.sha(accepted / n)
            == q.pb.qr.sha(old / "result" / n)
            == q.pb.qr.sha(frozen / n),
            "worst-plane bytes differ",
        )
    cmd = [
        sys.executable,
        "-B",
        str(q.HERE / "physical_batch.py"),
        "complete",
        "--case",
        "walnut720",
        str(accepted),
        r["report_sha256"],
    ]
    ref = subprocess.run(cmd, capture_output=True)
    q.need(ref.returncode == 3 and not ref.stdout, "queue may not complete missing cases")
    doc = dict(
        format="CFDK-physical-queue-walnut-qualification-v1",
        code_sha256=q.pb.qr.sha(q.__file__),
        harness_sha256=q.pb.qr.sha(__file__),
        frozen_walnut_sha256=BASELINE,
        case_commit_sha256=q.pb.qr.sha(state / "commits/walnut720.json"),
        accepted_report_sha256=r["report_sha256"],
        abandoned_report_sha256=old_hash,
        physical_worker_invocations=2,
        additional_worker_on_resume=0,
        verified_planes=len(orig["planes"]),
        verified_voxels=base["aggregate"]["voxels"],
        physical_cases=1,
        physical_specimens=1,
        external_complete_exit=ref.returncode,
        external_complete_stdout_bytes=len(ref.stdout),
        faults=expect,
        documents=results,
        scope="Real GB10 child and real process SIGKILL, not a power-loss or remote-filesystem guarantee. Two executions of one acquired walnut; pine cases pending an externally supplied parent.",
    )
    q.write(root / "qualification.json", doc)
    print(q.pb.qr.sha(root / "qualification.json"))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "fault":
        point = sys.argv[3]
        q.need(point in ("verified_before_commit", "committed"), "invalid fault")
        q.execute(
            sys.argv[2],
            work=True,
            after=lambda at, _case: os.kill(os.getpid(), signal.SIGKILL) if at == point else None,
        )
        raise RuntimeError("fault not reached")
    elif len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        raise ValueError("expected fresh run path")
