# Calibration-only control: can a one-voxel confidence shell correct near-cutoff pores?

Registered after E's development report, before any shell calibration results. Do not reuse a fitted full-calibration model, open a development evaluation stack, or read specimens 4 and 6. This control is not a new accepted candidate.

- Use development specimens 2, 3 and 5 only, all 24 calibration slices z=[160,184). Within the existing 256-wide crop, designate x=[0,96) and [160,256) as the two folds (64 unused columns in between). In each direction fit E with its existing settings on one fold only, then predict the other fold. Recompute weights on the training volume only; never use test labels, training side probabilities or coordinates for the correction. Keep the standard raw image halos. Count clipped objects separately in each fold, never join the two volumes artificially.
- Compare `p >= .5` with two separate, fixed alternatives. Select complete, six-connected **predicted** components of 32–160 voxels inclusive on the unmodified .5 mask. Expansion adds only one face-neighbour shell where `p >= .35`; contraction removes only the one-voxel inner face shell where `p < .65`. Apply each arm simultaneously, relabel after it, and allow any merge, split, false detection or loss to count. No changes to the 64-voxel measurement cutoff, match IoU or evaluation gates.
- Pool counts across the two disjoint folds **within** each specimen, not across specimens. Record all eligible reference and prediction counts, all formal matches, plus matched IDs, sizes and errors for the complete small references [64,128). For each arm compare the small reference IDs matched by both that arm and the baseline. Promote for further research only if the *same* arm on every specimen has at least five such pairs, reduces their P95 absolute relative volume error by at least 3 percentage points, and does not increase total missed references or total unmatched predictions. Insufficient pairs are inconclusive, never a pass. Publish all arms even if none is promoted. This does not estimate generalization or physical measurement accuracy.

<!-- generated shell calibration results -->

## Cross-fitted calibration results

| Specimen | Arm | Matched pores, base → arm / eligible references | Unmatched predictions, base → arm | Shared 64–127 voxel references | Shared-small volume error P95, base → arm | Decision |
|---|---|---:|---:|---:|---:|---|
| [2](evidence/shell-pilot/sample2.json) | expand | 8 → 8 / 9 | 1 → 1 | 5 | 17.97% → 18.20% | fail |
| [2](evidence/shell-pilot/sample2.json) | contract | 8 → 7 / 9 | 1 → 0 | 4 | 18.24% → 23.80% | insufficient pairs |
| [3](evidence/shell-pilot/sample3.json) | expand | 1 → 1 / 1 | 0 → 0 | 1 | 1.79% → 15.18% | insufficient pairs |
| [3](evidence/shell-pilot/sample3.json) | contract | 1 → 1 / 1 | 0 → 0 | 1 | 1.79% → 19.64% | insufficient pairs |
| [5](evidence/shell-pilot/sample5.json) | expand | 0 → 0 / 0 | 0 → 0 | 0 | — → — | insufficient pairs |
| [5](evidence/shell-pilot/sample5.json) | contract | 0 → 0 / 0 | 0 → 0 | 0 | — → — | insufficient pairs |

Neither arm meets the screening rule on all three specimens. Do not add a uniform confidence-shell correction to the production or research predictor on this evidence.

The denominator for each P95 is the shared subset in that row. A missing percentile means no common matched small pore, not zero error. The two test volumes are kept separate for connectivity; each fold trains on the opposite side only. These are calibration results and do not replace the [published E development evaluation](RESULTS-E.md).

## Was the split itself responsible for the scarcity of small pores?

Post-hoc count of the **entire original calibration crop**, before splitting. A component touching any of its six faces is clipped. A separate bounding-box implementation rereads the authenticated labels and checks the denominators of both folds for every arm.

| Specimen | Complete 32–63 voxels | Complete 64–127 | Complete ≥128 | Clipped ≥64 |
|---|---:|---:|---:|---:|
| 2 | 1 | 6 | 8 | 19 |
| 3 | 1 | 1 | 0 | 27 |
| 5 | 1 | 0 | 0 | 10 |

Specimen 5 has no complete eligible pore even before splitting, not an error-free small-pore measurement. The original labels still contain voxel-level boundary examples from clipped pores; those do not supply complete reference volumes. More independent, complete small-pore calibration examples are needed before selecting a size correction. Do not take them from the reserved confirmation regions.

[Full census and source hashes](evidence/shell-pilot/census.json). Recount with `python scripts/pore_calibration_census.py --cache <archive-directory> --records pores/evidence/shell-pilot --out pores/evidence/shell-pilot/census.json --check`.

Reproduce with `python scripts/pore_shell_pilot.py --cache <archive-directory> --out <empty-directory>` after compiling the CUDA kernels. `python scripts/pore_shell_report.py --check` rebuilds the summary table from the committed fold records, without reading a scan.
