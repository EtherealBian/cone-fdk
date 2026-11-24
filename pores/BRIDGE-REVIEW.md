# Exploratory review flag for possible merged components

This is a post-hoc diagnostic, motivated by seeing specimen 5's reference #42/#26 merge. The three transfer slabs have already been scored. This is **not** an independent test; it cannot promote a predictor or unlock a holdout. Lock this one rule before computing its flags and publish positives and misses on all three, with no search over thresholds.

On the sealed E probability map alone, label the six-connected p>=0.5 mask. For each complete predicted component of >=64 voxels, count disconnected six-connected p>=0.90 cores within it. Flag the parent for manual review when at least two cores each have >=32 voxels AND each has >=5% of its parent's volume. Use inclusive cutoffs and never split or change the prediction. No reference input when producing flags.

Afterwards define the narrow comparison event: one eligible parent covers at least half of **each of two or more different** complete reference objects of >=64 voxels. Report true/false flags, missed events, eligible negatives, IDs, core sizes and overlaps per specimen; do not count missing events as true negatives. This definition does not cover every possible merge. A flag is for review only, and an unflagged object is not cleared. No accuracy or time-saving claim.
