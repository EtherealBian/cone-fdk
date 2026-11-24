"""Render the registered shell-control decision from the committed per-fold records."""

import argparse
import json

from pore_shell_pilot import ROOT, summary

MARK = "<!-- generated shell calibration results -->"


def render(root):
    rows = []
    passed = {"expand": True, "contract": True}
    for i in (2, 3, 5):
        raw = json.loads((root / f"sample{i}.json").read_text(encoding="utf8"))
        if (
            raw["stage"] != "calibration only"
            or raw["specimen"] != i
            or set(raw["checked_folds"]) != {"L", "R"}
        ):
            raise ValueError("Unexpected shell experiment")
        out = summary(raw["checked_folds"])
        if out != raw["summary"]:
            raise ValueError("Summary does not replay")
        b = out["base"]
        for arm in passed:
            a = out[arm]
            passed[arm] = passed[arm] and a["decision"] == "pass"

            def pct(v):
                return "—" if v is None else f"{v:.2f}%"

            rows.append(
                f"| [{i}](evidence/shell-pilot/sample{i}.json) | {arm} | {b['matches']} → {a['matches']} / {b['eligible_reference']} | {b['unmatched']} → {a['unmatched']} | {a['small_shared_n']} | {pct(a['base_small_shared_p95_pct'])} → {pct(a['arm_small_shared_p95_pct'])} | {a['decision']} |"
            )
    selected = ", ".join(a for a, v in passed.items() if v)
    verdict = (
        f"Promote {selected} for study only."
        if selected
        else "Neither arm meets the screening rule on all three specimens. Do not add a uniform confidence-shell correction to the production or research predictor on this evidence."
    )
    counts = json.loads((root / "census.json").read_text(encoding="utf8"))
    census_rows = []
    for i in (2, 3, 5):
        c = counts[str(i)]["whole"]
        census_rows.append(
            f"| {i} | {c['complete_32_63']} | {c['complete_64_127']} | {c['complete_ge128']} | {c['clipped_ge64']} |"
        )
    return "\n".join(
        [
            MARK,
            "",
            "## Cross-fitted calibration results",
            "",
            "| Specimen | Arm | Matched pores, base → arm / eligible references | Unmatched predictions, base → arm | Shared 64–127 voxel references | Shared-small volume error P95, base → arm | Decision |",
            "|---|---|---:|---:|---:|---:|---|",
            *rows,
            "",
            verdict,
            "",
            "The denominator for each P95 is the shared subset in that row. A missing percentile means no common matched small pore, not zero error. The two test volumes are kept separate for connectivity; each fold trains on the opposite side only. These are calibration results and do not replace the [published E development evaluation](RESULTS-E.md).",
            "",
            "## Was the split itself responsible for the scarcity of small pores?",
            "",
            "Post-hoc count of the **entire original calibration crop**, before splitting. A component touching any of its six faces is clipped. A separate bounding-box implementation rereads the authenticated labels and checks the denominators of both folds for every arm.",
            "",
            "| Specimen | Complete 32–63 voxels | Complete 64–127 | Complete ≥128 | Clipped ≥64 |",
            "|---|---:|---:|---:|---:|",
            *census_rows,
            "",
            "Specimen 5 has no complete eligible pore even before splitting, not an error-free small-pore measurement. The original labels still contain voxel-level boundary examples from clipped pores; those do not supply complete reference volumes. More independent, complete small-pore calibration examples are needed before selecting a size correction. Do not take them from the reserved confirmation regions.",
            "",
            "[Full census and source hashes](evidence/shell-pilot/census.json). Recount with `python scripts/pore_calibration_census.py --cache <archive-directory> --records pores/evidence/shell-pilot --out pores/evidence/shell-pilot/census.json --check`.",
            "",
            "Reproduce with `python scripts/pore_shell_pilot.py --cache <archive-directory> --out <empty-directory>` after compiling the CUDA kernels. `python scripts/pore_shell_report.py --check` rebuilds the summary table from the committed fold records, without reading a scan.",
            "",
        ]
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p = ROOT / "pores/SHELL-PILOT.md"
    text = p.read_text(encoding="utf8")
    before = text.split(MARK)[0].rstrip() + "\n\n"
    s = before + render(ROOT / "pores/evidence/shell-pilot")
    if a.check:
        if text != s:
            raise SystemExit("Shell pilot report is stale")
    else:
        p.write_text(s, encoding="utf8")
