# Final accuracy gate — amendment after the sample 2 baseline, before inspecting samples 3/5

The user's tightened requirement supersedes protocol 1 as a release/stop criterion. Protocol 1 is retained unchanged as a baseline. A small error in total porosity alone can hide misplaced or missing pores. These are engineering targets versus the published digital segmentation, not certified dimensional tolerances.

Require ALL of the following on EACH confirmation specimen (3 and 5 for the baseline; reserve specimens 4 and 6 for a revised algorithm). Same registered interior crop, spacing and separate calibration slices as protocol 1. No test labels as pipeline input. No pooling, rounding up near misses or replacing a failed specimen. If a method is changed after seeing a test specimen, name it as development for the new method.

- Absolute error in ROI porosity <= 0.05 percentage points AND relative error <= 5% of the reference porosity.
- Voxel Dice >= 0.95, precision and recall each >= 0.93.
- Both sets of unclipped components with >=64 voxels: one-to-one IoU >=0.5 matches must cover >=90% of each set. Among matches: median absolute relative volume error <=5%, 95th percentile <=15%; median absolute relative equivalent diameter error <=2%; 95th percentile centroid displacement <=1.0 native voxel. An empty eligible set does not establish a pass.
- All exact same-mask, slab-invariance and per-ID audit requirements in protocol 1 still apply. Reference-assisted mask measurement is a distinct input mode and cannot pass the raw-to-mask accuracy gate. A failed candidate may be published as experimental evidence, never as the final accepted release.

## Operational stop rule (thresholds unchanged)

Do not score either reserved ROI until **all three** development specimens pass the unchanged thresholds, the development checks have been independently replayed against the source archives, and the candidate manifest has been published. Run the same frozen pipeline on **both** reserved specimens and replay both reports. A partial run, elapsed time, a good demo or agreement on a supplied mask is not a reason to declare completion. A failure after unblinding requires fresh confirmation specimens for any changed method. See the [freeze workflow](README.md#protect-the-confirmation-specimens).
