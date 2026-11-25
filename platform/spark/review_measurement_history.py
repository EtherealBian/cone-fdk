"""Compare complete original JUnit rosters; never turn any historical failure into a pass."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROSTER = "09a252e864cc6e7919962207d043fdbb9f28800f991600fd738c24553c89dc92"
PREVIOUS = "90ae24d1575f18a31449385d7e009ec430463f1e72e56bdce6eb72ccd248c3de"
CONTRACT = ["tests.test_axis::test_pinned_input_denies_writers_and_replacement"] + [
    f"tests.test_slab_replay::test_native_cannot_race_pinned_files[{i}]" for i in range(4)
]


def read(raw):
    cases = {}
    for t in ET.fromstring(raw).iter("testcase"):
        key = t.attrib["classname"] + "::" + t.attrib["name"]
        marks = [e.tag for e in t if e.tag in ("skipped", "error", "failure")]
        if key in cases or marks not in ([], ["failure"]):
            raise ValueError("duplicate, skipped or errored case")
        cases[key] = not marks
    if (
        len(cases) != 707
        or hashlib.sha256(("\n".join(sorted(cases)) + "\n").encode()).hexdigest() != ROSTER
    ):
        raise ValueError("changed test roster")
    return cases


def compare(old, new):
    if hashlib.sha256(old).hexdigest() != PREVIOUS:
        raise ValueError("previous JUnit digest")
    a, b = read(old), read(new)
    failed = lambda items: sorted(k for k, v in items.items() if not v)
    before, after = failed(a), failed(b)
    if len(before) != 10 or after != sorted(CONTRACT) or not set(after) <= set(before):
        raise ValueError(
            "unexpected regression or changed compatibility contract; inspect original XML"
        )
    return dict(
        format="CFDK-GB10-measurement-history-v1",
        roster_sha256=ROSTER,
        original_cases=len(b),
        previous_xml_sha256=PREVIOUS,
        current_xml_sha256=hashlib.sha256(new).hexdigest(),
        previous_passed=697,
        current_passed=sum(b.values()),
        current_failed=len(after),
        resolved=sorted(set(before) - set(after)),
        remaining_failed=after,
        new_failures=[],
        scope="Unfiltered original suite on the ordinary Spark filesystem. The five original-path lock assertions still fail; the Spark byte-snapshot contract has separate tests.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("previous", type=Path)
    p.add_argument("current", type=Path)
    a = p.parse_args()
    print(json.dumps(compare(a.previous.read_bytes(), a.current.read_bytes()), indent=2))
