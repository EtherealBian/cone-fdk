# Calibration check: which plane supports a threshold edge?

The paper describes a slice-wise operation. Test whether an axis mismatch could explain residuals before investing in a new model. Do not change E. Use the same 3-D median, then plain 5×5-patch/21×21-search CUDA NLM at scaled bandwidth 20, 30 or 40. Independently orient the NLM and the radius-5 Bernsen disk in XY, XZ or YZ. For each specimen, choose bandwidth for XY/XY and independently choose the best of the eight other plane pairs plus bandwidth on z=[160,164), X=[360,464); reapply both unchanged on z=[180,184), X=[512,616). Y=[376,632). The disjoint X ranges have a 48-pixel gap wider than twice the maximum preprocessing reach. Only raw halo voxels extend outside each labeled piece. Adopt an orientation change only if the *same* alternative plane pair wins in fit across all three specimens and each check improves Dice by ≥0.005. This tests an algorithmic proxy, not scanner orientation or physical truth.

## Development calibration results

| Specimen | Best alternative NLM / disk in fit | XY / XY check Dice | Alternative check Dice |
|---|---|---:|---:|
| [2](evidence/plane-pilot/sample2.json) | XY / XZ | 0.93647 | 0.71674 |
| [3](evidence/plane-pilot/sample3.json) | YZ / XY | 0.95448 | 0.90758 |
| [5](evidence/plane-pilot/sample5.json) | YZ / XY | 0.97631 | 0.95669 |

None improved on the held-apart calibration window. Keep the source XY plane for this filter and local threshold. There is no evidence here to justify treating the stack as misoriented. `python scripts/pore_plane_pilot.py --cache <source-directory> --out <empty-directory>` reproduces all 27 settings per scan.
