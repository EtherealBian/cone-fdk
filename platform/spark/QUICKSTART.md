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
- [Spark workspace regression](../../reports/gb10/spark-regression.json): See the linked receipt for the passing test count in the qualified workspace. Some tests and production qualification drivers require external acquisitions, retained caches and pinned prior jobs; this is not a claim that the whole suite runs on a standalone clone. The [periodic-only overlay](../../reports/gb10/historical-regression.json) passed 648 of 707 historical tests; the [slab compatibility overlay](../../reports/gb10/slab-compat-regression.json) passed 667, and the [directory edge overlay](../../reports/gb10/directory-edge-regression.json) passed 697 of the same 707, with no new failures.
- [Completed physical batch](../../reports/gb10/physical-batch-complete.json): both specimens and all three planned conditions. The [earlier walnut-only checkpoint](../../reports/gb10/physical-coverage.json) remains available. A solver comparison does not establish a physical measurement tolerance. Raw scans, scratch products and private execution logs are not part of this release.

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

<a id="offline-threshold-review"></a>
## Recount wall-loss footprints from the published masks (no GPU)

These **simulated** reference/inspection pairs use a 41 × 360 cylindrical grid. At each threshold pair the same two edge levels apply across the entire grid. The wall-loss gate stays at 0.3 mm. The repeat and noisy-change cases use 3 × 3 pairs; the invalid-edge control was archived with 2 × 2. Do not replace its missing observations with zero loss.

```sh
D=reports/gb10
python3 -I -S -B platform/spark/review_directory_masks.py \
  "$D/directory-edge-qualification.json" \
  39b156760dc467967d9372701843b392fd07c9347d35b5509d17d8da0723aa31 \
  "$D/directory-edge-masks.zlib"
```

Expected: **9 + 9 + 4 label grids, 324,720 cell decisions**. The 590-byte compressed input holds classified cells, not raw projections or precomputed component labels. A standard-library flood fill wraps across the angular seam but not the axial ends, assigns each region its first row-major cell ID, and checks the entire little-endian label grid against the pinned receipt. It also checks the unknown-cell, flagged-cell and region counts. It does not re-estimate wall thickness. [Raw CPU-to-CUDA checks and frozen file comparisons](../../reports/gb10/directory-edge-qualification.json) come from a separate Spark run with the original archived materials.

### Reopen a witnessed directory on the qualified Spark

After the default native `build.py` above (which creates the historical `.exe` names as symlinks to local ARM64 ELF products), use Python 3.12 with NumPy and SciPy. Build an optional module overlay in its own fresh directory. The destination's real parent directory must already exist.

```sh
D=$(mktemp -d)
python3 -B platform/spark/build_edge_compat.py "$D/overlay" \
  --expect 8ce35c3262a7d95de292616a5daa58655c41e60cd72710501a8ead7a78adeeb7
PYTHONPATH="$D/overlay:$PWD/scripts" python3 -B -m edge_review_packed --help
```

With your original witnessed job, its externally obtained manifest hash, and the original producer directory available, run:

```sh
PYTHONPATH="$D/overlay:$PWD/scripts" python3 -B -m edge_review_packed \
  "$PARENT_JOB" "$D/new-review" --levels 0.018 0.020 0.022 --budget 1 \
  --parent-sha256 "$PARENT_SHA256" --source-products "$ORIGINAL_PRODUCTS" \
  --companion-sha256 6351b4db3aa52e1df6a9651ccc329dcb238c45c6cdbc498dbd0c6cef9a72e7dc
```

Original producer files are authenticated as data and are not executed. Only the locally pinned ELF/PTX products run; `--source-products` can be omitted for a matching local producer. For a published result, exit 0 means complete coverage and 2 means the result contains unknown cells. Rejected inputs produce no delivery. This interface is for trusted local filesystems and synchronous consumers, not a distributed lock or power-loss guarantee. The full three-case CUDA qualification needs the archived source objects; only the offline mask check is self-contained. [Matched historical regression](../../reports/gb10/directory-edge-regression.json): 697 of the same 707 tests passed; the ten remaining failures are listed by name.

## Offline physical peak

Where do native FDK and ASTRA differ most on the measured walnut? This snapshot of the physical queue lets you check the answer without either solver. Use Python 3.10+ from the repository root, with `platform/spark/residual_witness.py` next to the reviewer. No installation, network, NumPy or GPU is required.

```sh
python3 -I -S -B platform/spark/review_physical_snapshot.py examples/physical-peak-walnut 98311760b93bf8636479d87b2b429a881b667ba649f987d2047c19dfada60370
```

![Maximum absolute native-to-ASTRA difference in each of the 512 axial planes](../../examples/physical-peak-walnut/profile.svg)

The reader reaggregates **512 reported plane rows**. It then reads **262,144 raw float32 pairs** from the plane with the highest peak. The first maximum in z, y, x traversal is at zero-based **(x=367, y=318, z=256)**: absolute intensity difference **0.000047359615564346313**. The chart uses every row, unsmoothed, with a linear intensity axis. To regenerate it, run `plot_physical_profile.py` with the same folder and pin, then `walnut720` and a **new** SVG filename. No raw voxels from the other 511 planes are in the pack. This measures agreement between implementations, not error against a calibrated specimen.

**Trust and coverage.** Obtain the catalog pin above from a trusted release, not from the submitted files. It binds the pack, case commit hashes and (if supplied) the external pine parent. The reader checks the frozen plan and parent receipts, case → committed report → plane hashes, volume lineage and reported totals before remeasuring the two slices. It does not refetch a scan, rerun either solver, or query the live queue. At this snapshot the walnut accounts for **1 of 2 specimens and 1 of 3 scan conditions**. Omitted cases have no metrics. `--require-complete` exits 3 without emitting a result; malformed or misbound data exit 1. A later submission does not change this snapshot.

**Export on the qualified Spark.** Pass the private queue and a fresh output directory to `platform/spark/pack_physical_snapshot.py`, with `--commit CASE COMMIT_SHA256` once for **every committed case**. Supply `--pine-parent SHA256` for committed pine cases. The producer invokes the original queue checks, authenticates the exact commit set under its lock, and rereads the detached result before publishing without replacement. Report hashes (including a correctly computed orphan) are not commit hashes. The fixed-order binary uses a magic header and bounded little-endian length-prefixed members, no executable code, extracted filenames or compression. [Measured export, refusals and comparison with the original walnut run](../../reports/gb10/physical-peak-snapshot.json) · [Acquisition and CC BY 4.0 attribution](../../NOTICE.md).

## Completed physical batch

This release adds a second **measured** specimen. The 90-view pine scan uses every eighth projection of the original 720. Its two conditions count as **one** pine cone. All volumes were compared at their original scale and on the same grid as their own ASTRA reference, not across specimens.

| Specimen | Views | Volume | Relative L2 (native vs ASTRA) | First maximum (x, y, z) | Absolute intensity difference |
|---|---:|---:|---:|---|---:|
| Walnut | 720 | 512³ | 0.0289893% | (367, 318, 256) | 0.00004735962 |
| Pine cone | 90 | 256³ | 0.0731422% | (87, 147, 235) | 0.00046228617 |
| Same pine cone | 720 | 256³ | 0.0265835% | (86, 100, 225) | 0.00015655230 |

```sh
python3 -I -S -B platform/spark/review_physical_snapshot.py examples/physical-peak-batch 95eb0678640814549ac9ba4e2b2b52b302e3bc38c759aae5186facdee95f7e31 --require-complete
```

Expected: **3 committed cases / 2 specimens**. The 3,674,999-byte pack contains 1,024 reported plane rows and three pairs of raw peak slices. On those slices the reader remeasures **393,216 voxel pairs**; it does not reread three full volumes. `plot_physical_profile.py` accepts `walnut720`, `pine90` or `pine720` from the same snapshot. Each plot scales its own vertical axis; use the table for cross-condition numbers.

**Source to result.** For the [pine acquisition](https://doi.org/10.5281/zenodo.6985407), the producer verified all eight archives and the metadata against the frozen publisher sizes/MD5, read frames 1–721 and excluded the duplicate 360° view. The 90 selected projections were independently preprocessed and matched the corresponding dense projections byte-for-byte. A second check reread the nine original files (4,225,994,738 bytes), both full native/ASTRA volume pairs, and all four direct native ROI crops. Both dense conditions passed the prespecified **relative L2 < 0.002** gate. No independently calibrated feature dimensions or defect labels are present. [Source manifest](../../fixtures/pine-cone-source-v1.json) · [Frame-to-archive hashes](../../examples/physical-peak-batch/pine-frames.json) · [Preprocessing receipt](../../examples/physical-peak-batch/pine-scan.json) · [Native/ASTRA parent](../../examples/physical-peak-batch/pine-parent.json) · [Independent recheck](../../reports/gb10/pine-independent.json) · [Final queue submissions](../../reports/gb10/physical-batch-complete.json).

<a name="spark-migration-acceptance"></a>

## DGX Spark validation

The main reconstruction and measurement jobs run on DGX Spark (GB10). The seven native build products, the 90-view native demo above, the two-specimen physical batch, and the archived axis and remount jobs have separate checks. Historical Windows binaries and the `measurement-lab` pore research branch are not part of the Linux runtime acceptance. The physical batch compares reconstruction intensities, not calibrated dimensions.

| Check on the qualified Spark | Result | Evidence |
|---|---|---|
| Spark-specific regression, with retained acquisitions and job products | 929 passed; no failures or skips | [Workspace receipt](../../reports/gb10/spark-regression.json) |
| Unmodified historical suite, same 707 test IDs | 702 passed, 5 failed; previously 697 passed. No skips or new failures. | [Resolved and remaining tests by name](../../reports/gb10/measurement-history.json) |
| Simulated tilted part, archived projections | All 25 sections measured; axis angle differed from design by 0.0017373°, origin by 0.0000292 mm. Independent check: 6,000 samples. | [Inputs and results](../../reports/gb10/measurement-jobs.json) |
| Simulated remount with a deep cut | Independent CPU labels agreed with CUDA on all 14,760 cells: 513 flagged, one region, none unknown. | [Coverage and matching archived files](../../reports/gb10/measurement-jobs.json) |

**Five historical failures remain.** One axis test and four slab cases require the *original pathname* to deny writes or replacement by another process. On the qualified Linux filesystem those names remain editable. Spark jobs consume sealed snapshots of the input bytes; another process cannot write through the snapshot, and changing the original name after sealing does not change the data being processed. These guarantees differ. The low-level axis `pinned_reader` refuses the old contract; code that depends on that contract must not rely on the Spark executable to deny writes to the original path. Use a trusted local job parent and a fresh destination. A duplicate remount exposure leaves only a refusal marker, never a success manifest.

### Run the measurement entry points

On the qualified ARM64 toolchain, start by running `platform/spark/build.py` as above. With Python 3.12, NumPy and SciPy, build the four overlays. Keep their import order: the measurement entry points call the original algorithms through the Spark-specific input and publication paths.

```sh
D=$(mktemp -d)
python3 -B platform/spark/build_publishers.py --out "$D/periodic"
python3 -B platform/spark/build_slab_compat.py "$D/slab" --expect f53f88ed79fe450ea502387d6152665ecd453c09a0efbbb76f7940e30dd7769a
python3 -B platform/spark/build_edge_compat.py "$D/edge" --expect 8ce35c3262a7d95de292616a5daa58655c41e60cd72710501a8ead7a78adeeb7
python3 -B platform/spark/build_measurement_compat.py "$D/measurement" --expect f98db1a6cd2099e1ffc1ca7e8b7c5fc5c347d1083e4bbc6adbee6d5378be6910
export PYTHONPATH="$D/measurement:$D/edge:$D/slab:$D/periodic:$PWD/scripts"
python3 -B -m axis_gauge --help
python3 -B -m remount_loss --help
python3 -B -m wall_pack --help
```

The Python `run(..., input_sha256={...})` interface accepts externally obtained SHA-256 pins for **every data input**. Omitting the map (including CLI runs) checks local self-consistency, not external provenance. Executable and PTX pins are always enforced. The simulated jobs in the table used external pins. Exact archive equality held for 5 of 8 axis outputs and 9 of 11 remount outputs; the independent numerical checks are reported separately. Full qualified regressions require the retained work products; the following release checks do not.

### Recompute the pine memory-budget comparison (no GPU)

```sh
python3 -I -S -B platform/spark/review_pine_budget.py reports/gb10/pine-budget-trials.zip examples/physical-peak-batch
```

This reads the 28 archived trials without executing the archived driver. Four warm-ups are excluded, leaving six paired trials at each view count. Raising the requested native buffer budget from 64 to 384 MiB changed the slab depth from 87 to 256 and reduced projection passes from three to one. At 720 views the median process times were **0.864 s and 0.659 s**, and the recorded peak owned allocations were **66,920,576 and 111,222,912 bytes**. Both budgets produced the same full volume. The budget is a cap, not measured memory use; process time includes startup and output and is not kernel time. [Trial-by-trial checks and summary](../../reports/gb10/pine-budget-review.json).
