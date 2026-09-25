# results/ — aggregate CSVs behind the paper tables
- `standardized_ppl.csv`: 512/256 token-weighted WikiText PPL (RUN_PPL_STD).
- `pruning_retention_05B.csv`: frozen V2 free×3 + perm for dense/s20 variants.
- `scale_retention.csv`: 1.5B + 3B matrix (PPL, free, perm, GSM).
Raw per-sample logs stay with the authors; manifests in `manifests/`.
