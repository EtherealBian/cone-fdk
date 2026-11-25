import json
from pathlib import Path

import pytest
from check_stripe_release import scale

FILE = Path(__file__).resolve().parents[1] / "reports/stripe-scale.json"


@pytest.mark.parametrize(
    "change",
    [
        "missing size",
        "memory policy",
        "missing oracle pair",
        "missing or reordered trial",
        "OS peak gate",
        "zero",
        "boolean",
        "OS growth gate",
        "work accounting",
    ],
)
def test_reject_false_scaling_claims(change):
    d = json.loads(FILE.read_bytes())
    big = d["cases"]["1536"]
    row = big["trials"][0]
    match = change
    if change == "missing size":
        del d["cases"]["384"]
    elif change == "memory policy":
        d["process_memory_gates"]["max_peak_mib"] = 1024
    elif change == "missing oracle pair":
        big["dense_oracle"].pop()
    elif change == "missing or reordered trial":
        big["trials"].reverse()
    elif change in ("OS peak gate", "zero", "boolean"):
        row["os_peak_working_set_bytes"] = {
            "OS peak gate": 256 << 20,
            "zero": 0,
            "boolean": True,
        }[change]
        match = "OS peak gate"
    elif change == "OS growth gate":
        row["os_peak_working_set_bytes"] = d["cases"]["384"]["trials"][0][
            "os_peak_working_set_bytes"
        ] + (48 << 20)
    else:
        row["payload_bytes_read"] -= 1
    with pytest.raises(RuntimeError, match=match):
        scale(d)


def test_os_variation_does_not_change_deterministic_review():
    d = json.loads(FILE.read_bytes())
    expected, peaks = scale(d)
    d["cases"]["1536"]["trials"][1]["os_peak_working_set_bytes"] += 4096
    now, other = scale(d)
    assert now == expected and other != peaks
