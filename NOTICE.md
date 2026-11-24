# Provenance and licensing

- Measured projections: Alexander Meaney, **Cone-Beam Computed Tomography Dataset of a Walnut**, University of Helsinki, [Zenodo record 6986012](https://doi.org/10.5281/zenodo.6986012), CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The distributed fixtures and previews are modified: uniform angular subsampling, optional five-pixel circular left shift, intensity binning, per-view air normalization, negative logarithm, and FDK reconstruction. The original data are already dark/flat corrected. The duplicate final view is omitted. No affiliation or endorsement is implied. See `fixtures/walnut-s8-b8/frames.json` for individual TIFF SHA-256.
- Geometry, metadata and rotation-center note: https://zenodo.org/records/6986012 . Background ROI convention: https://github.com/Diagonalizable/HelTomo .
- L. A. Feldkamp, L. C. Davis, J. W. Kress, “Practical cone-beam algorithm,” JOSA A 1, 612–619 (1984), doi:10.1364/JOSAA.1.000612.
- External reference only: [ASTRA Toolbox FDK_CUDA](https://astra-toolbox.com/docs/algs/FDK_CUDA.html), GPL-3.0. The native executable does not link to ASTRA and uses no ASTRA kernel. Comparative Python scripts use a separately installed ASTRA distribution.
- NVIDIA driver, NVRTC and cuFFT are external NVIDIA software subject to NVIDIA terms. No NVIDIA DLL is redistributed here. The core uses the CUDA Driver API (https://docs.nvidia.com/cuda/cuda-driver-api/) and cuFFT (https://docs.nvidia.com/cuda/cufft/).

- Measurement branch: Felix H. Kim et al., National Institute of Standards and Technology, [CoCr XCT data and author segmentation](https://doi.org/10.18434/M32162), [NIST data terms](https://www.nist.gov/open/license). On 2026-11-23 the images were cropped, filtered, segmented, measured and annotated for evaluation. No affiliation or endorsement is implied. No full scan archives are bundled. The published binary reference is a digital comparator, not traceable physical ground truth.

- On 2026-11-24 the NIST development scans were newly filtered and segmented (candidate D). The specimen-3 comparison shows a windowed, enlarged raw crop with digital-reference and C/D prediction overlays. Source: Felix H. Kim et al., National Institute of Standards and Technology, DOI 10.18434/M32162. No endorsement.

- On 2026-11-24 candidate E was evaluated on development specimens 2, 3 and 5 of the same NIST CoCr dataset. The offline review sheets contain contrast-windowed crops and C/D/E prediction overlays. These experimental measurements are not NIST-certified.

- On 2026-11-24 the boundary-transfer pilot cropped and annotated 96 additional slices each of development specimens 2, 3 and 5. Source: Felix H. Kim et al., NIST, DOI 10.18434/M32162. The review planes and specimen-5 plate use contrast-windowed raw data, digital-reference masks and experimental predictions; not a NIST-endorsed measurement.
