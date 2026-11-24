# Search support and bandwidth in the source-like NLM (development calibration)

After the E/F/G agreement diagnostic, test a common upstream assumption rather than voting. The ImageDev 2022.2 NonLocalMeansFilter2d documentation describes a square patch and square or disk search support, and explicitly defines radii rather than diameters. It is a related product, not evidence of which historical Avizo implementation, support or normalization produced the NIST data. Source: https://developer.imageviz.com/refmans/2022-2/ImageDev/html/Processing_ImageFiltering_SmoothingAndDenoising_NonLocalMeansFilter2d.html . NIST's published window **sizes** are 21 and 5. Keep the 5×5 patch and maximum shift of 10 unchanged.

Two prespecified arms: 21×21 square (441 offsets), closed radius-10 disk (317 offsets). Uniform patch SSD, self included, weight exp(-meanSSD/h²), FP64 sum of weighted intensities and normalization; leave median, digital mapping, local disk and strict threshold unchanged. Fit h in mapped intensity units over the integer grid 20..40 inclusive independently per arm and scan using maximum Dice (prefer smaller h on ties). The existing bandwidth 30 square is a separate fixed comparison; no scan-specific search shape mixing. Use development 2/3/5 only, x=[360,464), y=[376,632), z=[160,168) for fit, x=[512,616), same y, z=[176,184) for check, using 18 source halo pixels. Keep all 42 fit scores and 3 check masks' confusion counts. For each scan evaluate the fine square against the fixed square 30, then the fine disk against fine square. Require gain ≥0.005 Dice on every scan with precision and recall each no worse by more than 0.005. A winning calibration screen only motivates formal raw prediction; it neither certifies 3D pore accuracy nor permits opening the reserves. No formal annotation in this study.

## Held-apart calibration results

| Specimen | Selected h square / disk | Fixed square 30 Dice | Fine square Dice | Fine disk Dice |
|---|---:|---:|---:|---:|
| 2 | 27 / 29 | 0.94405 | 0.94093 | 0.92944 |
| 3 | 28 / 31 | 0.95829 | 0.95787 | 0.95740 |
| 5 | 33 / 36 | 0.98033 | 0.98097 | 0.97964 |

Neither comparison passes on all three scans. Do not promote either. [All fit scores, held-apart confusion counts and deltas](evidence/search-support.json). Reproduce with `scripts/pore_search_support.py` after building `search-support.ptx`.
