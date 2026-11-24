# Scoring I/O erratum (no change to masks or metrics)

The first invocation of the frozen scorer stopped at the first specimen, after producing its two new native tables but **before computing a comparison or a completed specimen report**. `pore_section_score.compare` raised `ValueError: Mismatched references`. JSON turns integer object IDs into strings, whereas freshly measured reference IDs are Python integers. The two inputs used different types for the same IDs. We retained the four files from the interrupted invocation locally and recorded their bytes below so they can be checked against the rerun. No result, metric, model, support or population was selected from this interruption.

The separately published reader `scripts/pore_monotone_typed_score.py` converts **only** the reference dictionary keys of the prior forward-control score files to integers, checking canonical spelling and bijectivity; it invokes the original scorer unchanged. Native outputs, matching rules, all thresholds and the six sealed predictions remain identical. Use a fresh output directory. This correction was published before the completed comparisons were run.
- `sample2-band-color-matches.csv`: `1a6b18a3e48c11d4ffa37240e7a92f8b7a698bbdda0ba6fa15d73d9dd9728a5d`
- `sample2-band-color-pores.csv`: `65f32657656ad9b80823d466e244134676fc837ec91617c37fcf4ac8e684a840`
- `sample2-band-sync-matches.csv`: `88b33807be319da710b57c9867e4bc1929ba3a3a88dac16d6a0440024b47da93`
- `sample2-band-sync-pores.csv`: `93273ba1b669c85931b16b5032602dab167bbf688cd714325f5a776a457eaf06`
