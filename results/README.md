# results/ — aggregate CSVs behind the paper tables
- `standardized_ppl.csv`: 512/256 token-weighted WikiText PPL (RUN_PPL_STD).
  Corpus: reproduce with `reproduce/prepare_wikitext.py` (sha 36d48636c19e).
- `behavior_scale.csv`: frozen V2 free + perm + fixed-extractor GSM.
  0.5B rows are HF-block (dense 16/77, s30 7/65); s30 GSM=4 is the C0 arm.
  0.5B dense GSM was never measured (empty cell, not zero).
  0.5B canonical pair lives in manifests (`RUN_05B_DENSE_HF.json`,
  `RUN_05B_S30_HF.json`, `RUN_05B_S30_FREE.json`).
- `standardized_ppl.csv`: 512/256 token-weighted WikiText PPL (RUN_PPL_STD).
  Corpus: reproduce with `reproduce/prepare_wikitext.py` (sha 36d48636c19e).
- `pruning_retention_05B.csv`: frozen V2 free×3 + perm for dense/s20 variants.
- `legacy/scale_retention_historical.csv`: 1.5B + 3B matrix (PPL, free, perm, GSM).
Raw per-sample logs stay with the authors; manifests in `manifests/`.
