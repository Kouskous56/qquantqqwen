# results/ — canonical aggregates
- `v3_cross_scale.csv`: CANONICAL cross-scale result (V3.3 typed rescore over
  frozen V3.1 raws). Regenerate: `python reproduce/rescore_v3.py ...` (see REPRODUCIBILITY).
- `standardized_ppl.csv`: canonical standardized WikiText perplexity.
  Corpus: `reproduce/prepare_wikitext.py` (sha 36d48636c19e).

# Historical / auxiliary (do not use for cross-block claims)
- `behavior_scale.csv`, `pruning_retention_05B.csv`: earlier V2/GGUF blocks.
- `legacy/`: superseded tables and runs.
