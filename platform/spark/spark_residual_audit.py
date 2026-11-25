"""Run the qualified physical walnut residual recheck under an explicit GB10 shared-memory budget.

The auto selector is calibrated for this exact device, stage, shape and sealed source pair.
It refuses an unqualified set or an insufficient cap before creating the result directory.
"""

import argparse
import json
from pathlib import Path
from mapped_transport import (
    BANDS,
    BUILD,
    HERE,
    MAPPED,
    MIB,
    ROOT,
    WALNUT,
    decision,
    held,
    json_checked,
    memory,
    need,
    run,
    sha,
    signature,
)
from spark_publication import fresh, publish, staging

PROFILE = "c32fe483e583bc43fc142999cc4d3b6d7005dd611c447e9a5e06f0f9fa1e3371"


def choose(summary, shape, cap_mib, band=None, policy="latency"):
    need(
        type(cap_mib) is int
        and 1 <= cap_mib <= 1024
        and (band is None or type(band) is int and band in BANDS)
        and policy in ("latency", "memory"),
        "invalid GB10 policy",
    )
    candidates = []
    for b in BANDS if band is None else (band,):
        s = summary[str(b)]
        preferred = s["latency_choice"]
        need(preferred in ("copy", "mapped"), "unknown winner")
        for mode in ("copy", "mapped"):
            size = memory(shape, b, mode)
            r = s["methods"][mode]
            need(size == r["combined_work_bytes"], "budget/measurement mismatch")
            if size <= cap_mib * MIB:
                # Respect the go/no-go latency gate at each band when both modes fit.
                if (
                    policy == "latency"
                    and mode != preferred
                    and memory(shape, b, preferred) <= cap_mib * MIB
                ):
                    continue
                rank = (size, r["median_ms"]) if policy == "memory" else (r["median_ms"], size)
                candidates.append((rank, b, mode))
    need(candidates, "no qualified configuration fits the cap")
    _, b, mode = min(candidates)
    return b, mode


def execute(out, cap_mib, band, policy):
    out = fresh(out)
    m = ROOT.parent
    prof = json_checked(m / "mapped-residual-bench-001/pairs.json", PROFILE)
    need(
        prof["format"] == "CFDK-GB10-residual-transport-pairs-v1"
        and prof["physical_qualification"] == WALNUT
        and prof["copy_build"] == BUILD
        and prof["mapped_build"] == MAPPED
        and prof["harness_sha256"] == sha(HERE / "mapped_transport.py"),
        "unqualified performance profile",
    )
    shape = prof["shape"]
    table = decision(prof["samples"], shape, prof["rows_sha256"])
    need(table == prof["summary"], "profile decision mismatch")
    b, mode = choose(table, shape, cap_mib, band, policy)
    with held("walnut720", m / "residual-walnut-001/qualification.json", WALNUT) as (
        source,
        exes,
        fds,
        expected,
    ):
        need(
            source["shape"] == [512] * 3 == shape
            and source["pins"] == prof["volume_sha256"]
            and expected == prof["rows_sha256"],
            "profile not valid for this source",
        )
        d = run(mode, exes[mode], fds, shape, b, cap_mib)
        sig = signature(d["rows"])
        need(sig == expected, "not the qualified residual rows")
        with staging(out, ".spark-audit-") as t:
            stage = Path(t) / "accepted"
            stage.mkdir()
            (stage / "native.json").write_text(json.dumps(d, indent=2) + "\n")
            result = dict(
                format="CFDK-GB10-profiled-residual-audit-v1",
                scope="Numerical QA of a previously authenticated physical walnut reconstruction pair. Not a defect/accuracy verdict.",
                policy=policy,
                requested_band=band,
                application_cap_mib=cap_mib,
                band=b,
                transport=mode,
                combined_work_bytes=d["combined_work_bytes"],
                profile_sha256=PROFILE,
                source_sha256=source["pins"],
                qualification_sha256=WALNUT,
                plane_witness_sha256=sig,
                verified_planes=len(d["rows"]),
                build_sha256=MAPPED if mode == "mapped" else BUILD,
                tool_sha256=sha(__file__),
                native_sha256=sha(stage / "native.json"),
            )
            (stage / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            publish(stage, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--cap-mib", type=int, default=96)
    p.add_argument("--band", type=int, choices=BANDS)
    p.add_argument("--policy", choices=("latency", "memory"), default="latency")
    a = p.parse_args()
    try:
        r = execute(a.out, a.cap_mib, a.band, a.policy)
        print(json.dumps(r))
    except (OSError, KeyError, ValueError, RuntimeError) as e:
        p.exit(1, f"Spark audit refused: {e}\n")
