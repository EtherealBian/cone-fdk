# Inverse patch-weight calibration: basis before normalization

**Development calibration screen, before running this estimator.** Voting did not remove shared errors. Moving a single Gaussian bandwidth or changing the search shape did not pass the separated checks. Instead learn the *shape* of a global nonincreasing patch-similarity curve within each scan. No specimen-coordinate features, spatial mask corrections, or labels outside the existing development calibration split. The failed vendor pixel controls are not calibration data.

Keep the median, full 21×21 search, square 5×5 patch, native source window, post-filter mapping and Bernsen decision fixed. Let d be patch RMS difference in published mapped gray units. Basis 0 is exp(-(d/30)^2); bases 1..6 are indicators d≤8,16,24,32,48,64 respectively. All include self. Form a convex combination of **unnormalized neighbor weights**, sum weighted intensities and normalize once per pixel; this is not an ensemble of independently normalized images. Use integer patch SSD and double accumulators for sufficient statistics before storing FP32 sums. Fit only one nonnegative seven-coefficient vector per scan, no spatial edits or variable decision threshold. Starting at pure Gaussian, for two rounds try every `(1-a)*current + a*one_hot(k)` for each of the six indicator bases, a=0.25,0.5,0.75,1; select highest exact calibration Dice, retaining current on ties then first enumerated proposal. Keep complete fit trace.

Development 2,3,5. Fit 8 slices z=[160,168), x=[360,464), y=[376,632); test the winning coefficients once on x=[512,616), same y, z=[176,184). Use 18 source halo pixels. Baseline is the same implementation with pure Gaussian. Both masks use the same strict threshold operators. Publish all three fit vectors, both confusion matrices, decisions and flip counts on the separated check. Advance to a new formal candidate protocol only if **all three** gain ≥0.005 check Dice without reducing precision or recall by >0.005. Otherwise archive the mechanism as a negative control, not a certified measurement. No formal or reserved region used in this calibration experiment.

## Results

| Specimen | Fit Dice: Gaussian → identified curve | Separated check Dice | Difference |
|---|---:|---:|---:|
| 2 | 0.91177 → 0.92562 | 0.94405 → 0.93890 | -0.00515 |
| 3 | 0.96040 → 0.96109 | 0.95829 → 0.95891 | +0.00062 |
| 5 | 0.97707 → 0.97707 | 0.98033 → 0.98033 | +0.00000 |

The shape did not pass the registered screen. For 5 the optimizer kept the original Gaussian. The actual weights and all 48 fit proposals for each scan are retained [here](evidence/inverse-weight.json). Do not promote or use any of the winning scan-specific changes in formal inference.

Numerical replay: at constant native gray 65535 the exact sum over 441 neighbors is 28,900,935. Storing the registered FP32 sum rounds it to 28,900,936; division by the exact mass yields 65535 + 1/441. The test checks this exact ABI result, its half-ULP/mass bound, and exact nearest-integer native gray rather than assuming exact real-valued normalization. No weights, masks, source intervals or accuracy gates were changed.
