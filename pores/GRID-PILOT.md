# Calibration diagnostic: is there a fixed patch lattice?

Compare whole-patch overlap-add on every pixel with estimates from a single source-XY lattice of stride 2, 3, 4 or 5 (all origin phases). Fixed 5×5 patches and 21×21 search on the 3-D median; bandwidths 20, 30, 40 in the scan's scaled intensity units. Pick the dense setting and sparse setting separately by Bernsen Dice on z=[160,164); compare these unchanged settings on [180,184). Only development 2, 3, 5. No evaluation slices. Require at least +0.005 check Dice on **each** scan to pursue a sparse lattice; otherwise keep dense estimators. This is a proxy and does not identify the proprietary implementation.

## Development calibration results

| Specimen | Chosen sparse stride / (Y,X) phase | Dense check Dice | Sparse check Dice | Difference |
|---|---|---:|---:|---:|
| [2](evidence/grid-pilot/sample2.json) | 2 / (1,1) | 0.92960 | 0.93386 | +0.00426 |
| [3](evidence/grid-pilot/sample3.json) | 2 / (0,1) | 0.95280 | 0.95071 | −0.00209 |
| [5](evidence/grid-pilot/sample5.json) | 2 / (0,1) | 0.97756 | 0.97658 | −0.00097 |

The required gain did not hold on any specimen. All fits chose bandwidth 30. Full rows are from the 24-pixel raw-context rerun (`e50a3e8`); extending the first run’s 18-pixel context did not change the selected methods or the displayed results. No sparse lattice in E.
