# Repository and local-data audit — 2026-10-08

Base: `489193e`. This audit refines runnable tools while preserving all original
experiment manifests, result CSVs, authored questions, and the frozen paper.

## Correctness changes

- Canonical V3 generation now uses the shared typed matcher, including ordered
  tuples. Rescoring rejects duplicate/missing IDs and incomplete repetitions.
  V2 and early V3 remain explicitly historical scoring protocols.
- PPL counts shifted causal labels and visits each target once. New results use
  `causal-token-weighted-v2`; old PPL scores are not silently rewritten.
- Wanda selects an exact number of weights even when importance scores tie.
  Recovery rejects fully truncated answer labels, checks restored mask shapes
  and layer names, and saves the tokenizer in both output directories.
- SparseGPT captures actual causal masks and rotary arguments from the model,
  propagates genuine model errors, and restores hooks/cache state. Mixed
  sliding/full attention is rejected instead of using the wrong mask.
- Qwen3 checkpoints identify the selected dataset, model name, arguments and
  source files; reject incompatible/corrupt state; recount saved labels; and
  save atomically per item. MMLU parsing handles lowercase and lone letters.
- The manifest patcher reads the published fallback map, refuses conflicting
  model hashes, and writes to a separate destination. Existing Qwen3 source
  snapshots retain the bytes referenced by frozen manifests.
- Dataset preparation, weight selection and output-directory handling are
  more explicit. Default pytest collection is restricted to `tests/`, avoiding
  accidental execution of historical experiment scripts under `legacy/`.

## Local evidence

Read-only inventory covered 606 project files (47,809,403,365 bytes) under the
lab root: 338 notes files, 115 scripts, 28 benchmark files, 68 report files,
16 reference PDFs, 36 model/Modelfile artifacts, and five root files. All
inventoried JSON parsed. Third-party tool trees and unrelated user directories
were excluded. Hashing a file establishes identity, not a substantive review
of every sentence in every report or reference PDF.

All 47 frozen Qwen3 manifests with an identifiable GGUF matched the current
local model bytes. Seven manifests still have no resolved model artifact.
The lab currently retains GGUF weights but not the original Hugging Face
checkpoint directories needed to replay canonical pruning/recovery/PPL.

The local release copy contains byte-identical authored questions, the six
canonical V3.1 raw-generation manifests and `results/v3_cross_scale.csv`.
Other local-release file differences include line endings and historical
provenance versions; a byte mismatch alone is not evidence of score drift.

Four local replication files, absent from the public frozen manifest set,
cover question IDs 400–799:

- `RUN_GSMSWEEP_Q4KSR_400.json` and `RUN_GSMSWEEP_Q4KSR_600.json`
- `RUN_GSMSWEEP_Q4KMR_400.json` and `RUN_GSMSWEEP_Q4KMR_600.json`

Pairing by question ID and verifying the stored gold/extracted answer labels
gives KS 270/400 versus KM 276/400; n11=255, n10=15, n01=21, n00=109;
two-sided exact McNemar p=0.40503224614076316. This supports the documented
failure to replicate the earlier advantage; it does not prove equivalence or
the proposed mechanism. The audit report stores input SHA-256 values without
publishing the local response texts.

The local `RUN_PPL_STD.json` records the old 12,800-token denominator. Historical
GSM audit logs also contain extraction failures (for example a final boxed 18
reported as a dot), so generation quality and extraction quality must be
distinguished. New raw responses remain available for subsequent rescoring.

## Validation and limits

Regression coverage includes frozen V3.3 totals, Qwen3 manifest integrity,
interrupted/resumed evaluation, invalid checkpoints, atomic writes, paired
question matching, PPL window accounting, tied pruning metrics, and masked
recovery. A tiny randomly initialized Qwen2 model tests SparseGPT adapter
forward preservation on CPU without downloading weights.

This is not a full GPU rerun. The original model checkpoints are missing;
historical seeds and some original scorer versions are unavailable. The
standalone systems/IFEval scripts under `bench/` and `legacy/` remain lab-era
artifacts and are not promoted to canonical portable runners. In particular,
`measure_11a.py`'s text-stream timing is not a verified first-token benchmark,
and the old IFEval resume files do not identify the dataset/model configuration.
Use their frozen results as historical observations with those limitations.

Reconciliation commands are in `REPRODUCIBILITY.md`. They read local source
data and write a separate report; they do not modify or publish the lab files.
