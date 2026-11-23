# Predeclared acceptance gates for full-circle FDK

All amplitudes are in mm^-1. No gain fitting, image alignment or masking when comparing to ASTRA. The 180-view analytic phantom and real scans use different source distances and detector sizes.

- Analytic finite cylinder with two eccentric holes and an off-plane inclusion: bulk attenuation within 3% of 0.04/mm; both hole centers within 0.4 mm; both equivalent diameters within 0.6 mm; positive contrast at known off-plane inclusion. Exterior background |mean| < 0.0015/mm.
- Matched ASTRA 2.5 `FDK_CUDA` Ram-Lak: full-volume relative L2 < 1% on analytic 256³ and 512³, and < 2% on real binned walnut. No fitted intensity scale, axis flips, rotation or postprocessing.
- Forced memory constraint: actual multi-slab run, owned CUDA allocations (including cuFFT work area) at or below budget, and rel-L2 <= 1e-5 vs unrestricted; tail view and tail slab exercised. Identical repeat input and settings yield deterministic volume bytes.
- Real data: full 360° coverage using uniformly sampled measured views, excluding the 360° duplicate; report exact number, binning, correction, provenance. Deliver corrected/uncorrected slice pair and an offline fixture. No metrology accuracy claim on the walnut.
- Same PC benchmark: 256³ and 512³ at fixed views, wall and GPU event stage timings, memory and GPU baseline. No speedup requirement against ASTRA. Repeat runs; report medians without mixing timings from different timing boundaries.
- Reject malformed geometry, missing/short/long/nonfinite projection files, impossible budgets and aliased output; failed runs do not replace an existing output. Validate test results on the real GPU.

## Native optimization release gate (v1 baseline vs candidate)

Before recording the release benchmark, freeze both EXE/PTX pairs. On this same PC run the 180-view phantom and 720-view corrected walnut at 512³, 768 MiB, batch 16. Warm each binary once per case; then run six paired trials, alternating AB and BA, including file flush and replacement. For **each** input, candidate median native wall time **and** outer process time must each fall by at least 5%, with wins in at least five of six pairs for each metric. Every output (including warmups) must have identical full SHA-256; no increase in engine-owned device bytes. Keep all trials. The existing full correctness, oversize and packaging gates still apply. This is a local performance qualification, not a cross-machine guarantee.
