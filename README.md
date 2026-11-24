# ConeFDK · Single-GPU cone-beam CT reconstruction

**Experimental branch:** [CoCr pore measurement, accuracy gates and evidence](pores/README.md). [D → E: same-pore errors, misses and three offline review packs](pores/RESULTS-E.md). [Cross-specimen boundary transfer: three new slabs, per-pore counts and a documented merge](pores/RESULTS-TRANSFER.md). The stable walnut reconstruction release is unchanged.

Reconstruct a 3D attenuation volume from a full rotation of X-ray projections on a Windows PC. ConeFDK uses native C++ and CUDA for cone weighting, filtering and backprojection; cuFFT provides the batched 1D transforms. Python prepares input and checks the result against independent implementations. It is not needed in the reconstruction executable.

[中文文档](README.zh-CN.md) · [Windows release](https://github.com/EtherealBian/cone-fdk/releases/tag/v0.1.0) · [Download the offline slice viewer](https://github.com/EtherealBian/cone-fdk/releases/download/v0.1.0/walnut-512-viewer.html)

![Axial sweep through a walnut reconstructed from 720 measured views](reports/walnut-sweep.gif)

*The image is a sweep through measured data, not a rendering of a synthetic walnut. The downloadable HTML lets you scrub 81 preview slices without installing CUDA. Data: [Alexander Meaney / University of Helsinki](https://doi.org/10.5281/zenodo.6986012), CC BY 4.0; processing and attribution in [NOTICE](NOTICE.md).*

**Measurement lab (experimental):** [Calibrated intensity model G, raw-weight replay and shared segmentation errors](pores/RESULTS-G.md) · [prospective false-split counterexample](pores/RESULTS-F.md). No cross-specimen acceptance claim.

## What was measured

| Question | Result on one RTX 5060 Laptop GPU | Evidence |
|---|---|---|
| Does the reconstruction agree with an independent implementation? | 720 measured views, 8×8 detector binning, 512³ FP32 output. Full-volume relative L2 difference from ASTRA FDK: **0.02899%**. No registration or fitted intensity scale. | [Numerical checks](reports/RESULTS.md) |
| Did the optimization help the whole application? | Median process time **2.392 → 2.009 s**, including startup and output flush (−16.0%); 6/6 paired runs faster, identical output bytes. Compared with the previous native engine, **not** ASTRA. | [All A/B trials](reports/native-ab.json) |
| Can it reconstruct a volume larger than VRAM? | Wrote and hashed a **1,296³ FP32** volume (8,707,129,344 bytes) with a **512 MiB engine-owned device-buffer budget**. Five slices checked against ASTRA; not a full-volume cross-check at this size. | [Stress record](reports/oversize.json) |

## Try it

- **Just inspect the result:** download the HTML viewer above and open it in a browser. No installation or network connection required to view the slices.
- **Run the offline 90-view demo:** extract `ConeFDK-win-x64.zip` from the release. In that directory run the commands below on a compatible Windows x64 NVIDIA machine. The release includes the executable, PTX and measured fixture, not NVIDIA DLLs. Installing dependencies requires internet access.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win.lock
.\Demo.ps1
```

- **Build from a source clone:** install the same locked dependencies, open an x64 MSVC Developer PowerShell (or set `CXX` to a 64-bit MinGW compiler), then run `.\Build.ps1` before `.\Demo.ps1`. `.\Run-Acceptance.ps1` also exercises the 720-view dataset and the disk-backed stress case. Allow about 25 GiB free. See the [input contract and exact CLI examples](README.zh-CN.md#自有扫描如何接入).
- **Reproduce the optimization comparison:** the release includes a compact `ConeFDK-ab-baseline.zip` with the previous native executable, PTX, source and reference output hashes. See [benchmark protocol](reports/OPTIMIZATION.md). Its wrapper is a new ZIP, not the archived full development release; the native binary hashes match the original benchmark.

## Where the engineering is

| Read | Why |
|---|---|
| [CUDA kernels](src/kernels.cu) | Voxel-driven backprojection, Z-register tiling and interpolation coefficients packed into the former FFT scratch region. |
| [Memory planner and runtime](src/main.cpp) | Selects angle batches and Z slabs under a byte budget. Smaller budgets trade rereading/refiltering for lower residency. |
| [Windows file I/O](src/files.hpp) | Unicode paths, input/output alias checks, and transactional publishing after successful validation and flush. |
| [FP64 spatial oracle](tests/test_spatial_oracle.py) · [Acceptance gates](reports/ACCEPTANCE.md) | Direct convolution and independently constructed projection geometry, alongside analytic phantoms and real data. |

**Scope:** centered, circular, full-scan cone-beam geometry with a flat detector. Experimental reconstruction software, not a calibrated industrial inspection system or clinical device. Agreement with ASTRA is not dimensional ground truth. The reported memory budget excludes driver/context allocations. No claim that Compute Sanitizer passed; its attachment limitation is documented in the [test report](reports/RESULTS.md).

Original code: [MIT](LICENSE). Data and external tools retain their own licenses; see [NOTICE](NOTICE.md).
