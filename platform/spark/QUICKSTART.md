# Run the Spark walnut example

## Inspect two real regions without CUDA

From the repository root, run this with Python 3.12. It reads the paired FP32 subvolumes directly from the checkout. No downloads, NumPy, ASTRA, GPU or device access are needed.

```sh
D=examples/gb10-walnut
python3 -I -S -B platform/spark/review_grid_oracle.py "$D/qualification.json" \
  445b3a5ad5a6585b9621a7d57ca6fb1661e403b27714613e997907449cf730ac \
  "$D/native.mg" "$D/astra.mg"
python3 -I -S -B "$D/test_review.py"
```

| Parent grid | Region (X × Y × Z) | Voxels checked | Relative L2 | Maximum absolute difference |
|---|---|---:|---:|---:|
| 225³ | 39 × 53 × 31 | 64,077 | 0.000230125 | 0.0000539310 |
| 384³ | 47 × 59 × 27 | 74,871 | 0.000338438 | 0.0000453368 |

These are two regions of the **same 720-view walnut acquisition**. The verifier checks grid headers, pins, layout, per-plane statistics and the global error gates. The four public tests include a changed voxel, swapped arrays and an incorrect receipt hash. This checks the committed data; it does not rerun ASTRA. See the [request](../../examples/gb10-walnut/request.json) and [data attribution](../../NOTICE.md).

## Run native CUDA from a fresh source checkout

Tested on Linux ARM64, NVIDIA GB10, CUDA 13.0.88 and g++ 13.3. Use a fresh checkout with no `build/` directory. The native executable loads the CUDA driver and cuFFT (`libcufft.so.12`); ASTRA is not linked. NumPy is needed only to unpack the included 90-view fixture for this example. The multi-grid qualification builder intentionally refuses a base product that differs from its pinned compiler/toolkit output. Such a refusal is not a numerical result.

```sh
python3 -B platform/spark/build.py
python3 -B platform/spark/build_multigrid.py build/multigrid
python3 -B examples/gb10-walnut/native_demo.py
```

The native demo reads the 90-view subset shipped with the repository. It reconstructs two grids with a single projection pass, then separately reconstructs each ROI and compares its full byte hash. The successful output includes `matched_standalone: 2`. This smaller example is **not** the 720-view timing benchmark or an independent solver check.

## What the release establishes

- [Full walnut comparison](../../reports/gb10/walnut-astra.json): 134,217,728 voxels; no registration or fitted intensity scale. [Two native repeats](../../reports/gb10/walnut-native.json) produced identical bytes.
- [Three-grid timing](../../reports/gb10/multigrid-timing.json): 192³ overview, 512-grid ROI and 1024-grid detail. Six warm pairs, 3 → 1 projection reads and 135 → 45 filter batches. Median native process time includes startup and output sync, excludes sealing, verification and export.
- [Independent ASTRA grid audit](../../reports/gb10/grid-agreement.json): 12,618,525 voxels on the three preset grids. [Host-buffer comparison](../../reports/gb10/astra-buffer-timing.json): six interleaved pairs, unchanged output, median pipeline 650.894 → 545.263 ms. These are not device-memory savings or native FDK timings.
- [Spark workspace regression](../../reports/gb10/spark-regression.json): 809 passed in the qualified workspace. Some tests and production qualification drivers require external acquisitions, retained caches and pinned prior jobs; this is not a claim that the whole suite runs on a standalone clone. The [periodic-only overlay](../../reports/gb10/historical-regression.json) passed 648 of 707 historical tests; the [slab compatibility overlay](../../reports/gb10/slab-compat-regression.json) passed 667 of the same 707, with no new failures.
- [Physical batch](../../reports/gb10/physical-coverage.json): one of two planned specimens in this snapshot. A solver comparison does not establish a physical measurement tolerance. Raw scans, scratch products and private execution logs are not part of this release.

## Resume a five-slab job without changing a legacy callback

On the qualified DGX Spark, a 720-view walnut reconstruction was interrupted with `SIGKILL` after two durable slab commits. No final delivery existed. A synchronous callback using the original path-based CUDA command resumed the remaining three slabs. All 536,870,912 output bytes matched the retained dense volume; the original verifier reopened the delivery with only the Python standard library. See the [measured recovery receipt](../../reports/gb10/slab-recovery.json).

To check the compatibility layer without an acquisition or GPU job, start at the checkout root on ARM64 Linux with Python 3.12 and pytest:

```sh
python3 -B -m pytest -q -p no:cacheprovider platform/spark/tests/test_slab_compat.py
```

For an integration, build the module in a fresh directory, separate from the periodic overlay. Import it before the historical `scripts/` directory:

```sh
D=$(mktemp -d)
python3 -B platform/spark/build_slab_compat.py "$D/overlay" \
  --expect f53f88ed79fe450ea502387d6152665ecd453c09a0efbbb76f7940e30dd7769a
PYTHONPATH="$D/overlay:$PWD/scripts" python3 -B -c 'import slab_replay as s; print(s.__file__)'
```

Native jobs require the exact plan, acquisition, ELF and PTX pins. The default worker inherits sealed file descriptors. A custom callback gets temporary paths to the same sealed bytes and can run a synchronous child; it must not save these names for later. The original names may be renamed or modified on Linux. A new invocation rejects a changed source even with a completed checkpoint. This checks process recovery on trusted local storage, not recovery after power loss. The [historical comparison](../../reports/gb10/slab-compat-regression.json) lists every remaining failure, including four tests that require Windows-style denial of original-path writes.
