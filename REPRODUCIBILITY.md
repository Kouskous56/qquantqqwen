# REPRODUCIBILITY — claim → command → manifest → expected output
Models (download once from Hugging Face): Qwen2.5-{0.5B,1.5B,3B}-Instruct.
Set MODELROOT to your weights directory. Model evaluation/training uses CUDA
unless a CPU option is specified; rescoring and integrity tests need no GPU.

## Installation and offline checks

New mathematics runs use the separately frozen [Math50 V4 protocol](data/v4_0_1/README.md).
Export its 50 free or 200 rotated MCQ prompts with `python -m reproduce.eval_suite_v401
--mode free --prepare-only --out runs/v4/prompts.json`. Historical commands below
retain their original evidentiary meaning; V4 scores must not be substituted into
the frozen tables. [The migration report](results/v4_0_1_migration/MIGRATION.md) explains
scorer-only differences and excludes changed stems from answer-key migration.

Use Python 3.11 in a virtual environment. For integrity checks only:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

For model runs, install the PyTorch build appropriate for your hardware first,
then `python -m pip install -r requirements.txt`. The historical CUDA 12.9 lab
used `torch==2.9.0+cu129` from `https://download.pytorch.org/whl/cu129`.
`requirements-lock.txt` is an environment snapshot, including optional tools,
not a portable lock for all platforms. GGUF V2 needs `llama-cpp-python`;
IFEval needs `lm_eval[hf]` and its task dependencies. Lightweight tests skip
Torch-specific checks when Torch is absent; CI also runs a CPU Torch job.

Run new experiments into separate output directories. Original manifests,
aggregates and the frozen paper are evidence, not destinations for new runs.

## Standardized PPL (historical claim: ΔPPL@s30 ≈ +6–8% all scales)
```bash
python reproduce/eval_ppl.py --model $MODELROOT/Qwen2.5-0.5B-FP16 \
  --tok $MODELROOT/Qwen2.5-0.5B-FP16 --corpus bench-corpus/wikitext_test.txt \
  --out results/ppl_05B_s0.json
```
Historical recorded results: `05B-s0 ≈ 15.11`, `05B-s30 ≈ 16.31`, `15B-s0 ≈ 9.93`,
`15B-s30 ≈ 10.58`, `3B-s0 ≈ 8.76`, `3B-s30 ≈ 9.31` (see `results/standardized_ppl.csv`).
These are not expected outputs of the corrected evaluator. The earlier code
overcounted causally shifted labels and could score a final window twice.
New runs identify `protocol: causal-token-weighted-v2`, score each token after
the first exactly once, and report `tokens = input_tokens - 1`. The historical
numbers have not been replaced or corrected by estimation; rerun both arms
with the same corpus and corrected protocol before comparing new PPL values.
Prepare that corpus with `python reproduce/prepare_wikitext.py --help`.

## V2 behavioral eval - quantization agreement (legacy GGUF block, historical)
```bash
python reproduce/eval_v2.py --model model-q4.gguf --questions data/v2/questions.json --use-template --out results/v2_3b.json
```
Expected (3B self-Q4, historical): perm about 149/200, free about 37/50.

## V3.3 unified cross-scale pruning (canonical current)
Generate once per arm (3 reps), then rescore frozen raws (no GPU needed):
```bash
python reproduce/eval_v3b.py --model $MODELROOT/<dense-or-pruned> --questions data/v2/questions.json --out results/v3.json --tag <arm>
python reproduce/rescore_v3.py --manifest-dir manifests --questions data/v2/questions.json --out-results results/v3_cross_scale.csv --out-manifest manifests/RUN_V33_RESCORE.json
```
Expected: 0.5B 18→11, 1.5B 40→33, 3B 42→40 — i.e. **−7 / −7 / −2 items**.
Quote the counts, not the retention ratios (61% / 82.5% / 95%): the ratios differ
in part because the three scales start from different baselines (18 / 40 / 42).
At 0.5B, 10 dense-correct items become wrong and 3 dense-wrong items become right.
The net loss is 7; score retention is not conditional item retention.
Manifests: RUN_V31_* (frozen raws) + RUN_V33_RESCORE.json. Do not mix GGUF-block and HF-block numbers.

## GSM8K held-out (Claim: recovery is domain-specific)
```bash
python reproduce/eval_gsm.py --model $MODELROOT/<pruned-or-recovered> --n 50 --out results/gsm.json
```
Expected: s30-0.5B base 4/50 (sparse), GSM-masked-recovery 14/50.

## Wanda pruning (Claim: s20 safe, s30 boundary at 0.5B)
```bash
python reproduce/prune_wanda.py --src $MODELROOT/Qwen2.5-0.5B-FP16 \
  --out $MODELROOT/pruned-s30 --sparsity 0.3
```

## Masked recovery (Claim: sparsity preserved + domain recovery)
```bash
python reproduce/recover_masked.py --src $MODELROOT/pruned-s30 --data gsm8k \
  --out-merged $MODELROOT/rec-merged --out-masked $MODELROOT/rec-masked
```
Expected: merged sparsity → 0.00, masked stays 0.30.

## Manifests
Provenance coverage varies by historical manifest — see `manifests/`.
Legacy development scripts live in `legacy/`; canonical entry points are `reproduce/`.

---

# Qwen3-4B block (`manifests_qwen3/`)

Different generation from the Qwen2.5 block above, on purpose. Read this before
comparing any number across the two.

**What is actually reproducible here, and what is not.** Every Qwen3 score was
produced by running a 2.5–3.3 GB GGUF through Ollama on one RTX 4050 (6 GB).
That does not fit in CI. So for this block CI verifies *integrity*, not
*regeneration*:

| What CI checks | What CI cannot check |
|---|---|
| every required field present | that a score matches the model |
| hashes are 64-hex and well-formed | that the GGUF still exists on your disk |
| `dataset_sha256` recomputed from `details` | that the labels were scored correctly |
| `scores` agree with a recount of `details` | — |
| recorded source hash matches retained source bytes, when available | unavailable run-time source versions |
| optional original-manifest hash comparison with `QWEN_LAB_NOTES` set | current model bytes without local access |

Every Qwen3 manifest therefore carries `regenerable: false` plus a `limitation`
string saying so. That field is a constant, not an oversight: the scores are
real and their provenance chain is checked, but re-obtaining them needs the
GPU.

**Provenance chain.** `model` → Ollama `FROM` blob → the blob's filename *is* the
GGUF's SHA-256 → the file in `models/`. `provenance.gguf_resolved_via` records
which link was used. Where a manifest already named its GGUF, the hash is
compared against the original and the result is in
`provenance.gguf_sha256_matches_original` (10/10 match, 0 mismatch).

**Where a manifest cannot be verified, it says so instead of guessing:**

- 7 of 54 have no resolvable artifact (`gguf_available: false`) — pruned FP16
  directories were deleted during cleanup, so there is nothing left to hash.
- 12 of 54 have no identifiable scoring script
  (`scorer_status: khong-xac-dinh-duoc`, `scorer: null`): 6 `RUN_QWEN3_TYPED_*`,
  5 R1 MMLU (their record shape matches no script on disk), 1 smoke probe.
- `timestamp` is the manifest's file mtime, not the original run time. This is
  stated in `timestamp_source`. File mtimes are not preserved by git, so treat
  these dates as unreliable.

**`dataset_sha256` means different things per schema**, which is why
`dataset_sha256_scope` is mandatory and `detail_schema` is one of four values:

| `detail_schema` | hash covers | can it prove the dataset is unchanged? |
|---|---|---|
| `mmlu` | (i, subject, gold) | no: question text and choices are not covered |
| `gsm` | (i, expected answer) | no: question text is not covered |
| `typed` | (id, type, raw) — rescore over frozen generations | yes, for the raws |
| `harness` | **(i) only** | **no** |

The `harness` case is the weak one. `RUN_HARNESS_F16*` recorded item index and
pass/fail but not the expected answer, so its hash pins *which 400 items ran* and
nothing about their content. Stated plainly rather than left for a reader to
discover.

## Re-running a Qwen3 measurement yourself

Requires: Ollama, a Qwen3-4B GGUF, and the same calibration file. The seven
scoring and provenance scripts are in `bench/qwen3/` (`gsm_sweep.py`,
`mmlu_any.py`, `harness_probe.py`, `eval_round2.py`, `gsm200_ollama.py`,
`eval_qwen3_gsm.py`, plus `patch_manifests.py` which rebuilds provenance
without touching scores).

```bash
# GSM8K, CoT, greedy. Two blocks of 200 so a mid-run failure loses one block.
python bench/qwen3/gsm_sweep.py --model <ollama-tag> --tag MINE --precision Q4_K_M \
  --n 200 --offset 0
python bench/qwen3/gsm_sweep.py --model <ollama-tag> --tag MINE --precision Q4_K_M \
  --n 200 --offset 200

# MMLU, forced single-letter, npredict 32
python bench/qwen3/mmlu_any.py --model <ollama-tag> --tag MINE --precision Q4_K_M
```

Both checkpoint per item and resume, so killing one does not lose the run.
The Ollama runners accept `--notes-dir` (default `D:/qwen/notes`) and
`--endpoint` (default `http://127.0.0.1:11434`). Use a fresh tag and a new
output directory for corrected runs. The FP16 runner instead accepts
`--model-root`, `--notes-dir` and `--n`.

New checkpoints include complete selected dataset content, source hashes and
generation settings in their identity. Resume verifies every saved response
and label and rejects corrupt, incompatible or legacy checkpoints. Historical
checkpoints lack that identity; retain them and use a fresh tag for a new run.
Writes are atomic, so an interrupted save preserves the previous checkpoint.

The pre-refinement Qwen3 source files are retained in `bench/qwen3/frozen/`.
Frozen manifests keep their original `scorer_sha256` and `current_sha256`;
the latter means current **when that manifest was patched**, not current HEAD.
Tests resolve those recorded digests to retained source bytes. This does not
recover source versions that were already missing at the time of the audit.

**One number from this block needs a caveat, and it is not a small one:** a
single 200-question block is not a stable unit. A quantization effect of ~4pp
gave McNemar p=0.019 in block A and p=0.424 in block B on identical models and
identical protocol. Pooled over 400 questions: −4.00pp, p=0.0226. If you are
reading a sub-5pp difference from a single 200-item run here, it is not
established.

## Regenerating the patched manifests

`bench/qwen3/patch_manifests.py` rebuilds `manifests_qwen3/` from the lab's raw
`notes/` directory. It only adds provenance and never edits a score. It is not
needed to use this repo; it is here so the transformation is auditable rather
than something that happened once.

Pass `--notes-dir`, `--models-dir` and a separate `--output-dir` explicitly.
The patcher rejects writes over this checkout's frozen manifests and rejects
an artifact whose bytes conflict with the original recorded model hash.

## Read-only reconciliation with the local lab

```bash
python reproduce/audit_local_data.py --lab-root D:/qwen --release-root D:/qwen_release --hash-models --out ../local-data-audit.json
python reproduce/compare_gsm_runs.py --arm-a D:/qwen/notes/RUN_GSMSWEEP_Q4KSR_400.json D:/qwen/notes/RUN_GSMSWEEP_Q4KSR_600.json --arm-b D:/qwen/notes/RUN_GSMSWEEP_Q4KMR_400.json D:/qwen/notes/RUN_GSMSWEEP_Q4KMR_600.json --out ../q4ks-replication.json
```

The inventory hashes project data, lists models, validates JSON, and optionally
rehashes the large GGUFs. It excludes third-party tools. It writes no source
files and copies no raw responses into its report. The comparison checks paired
question IDs, gold answers, block completeness and stored labels before computing
the exact two-sided McNemar p-value. Source hashes identify the inputs; this is
a recount of saved measurements, not a fresh model evaluation.


## Audit corrections (2026-10-09)

Use Math50 **4.0.1** for new runs. Version 4.0.0 is retained for audit and has a
known symbolic false positive. Shared Qwen3 checkpoints now require artifact
identity version 2; old checkpoints must use a new run/tag rather than being
silently resumed. All shared Ollama runners accept `--gguf`, and remote servers
require it. GSM `--scoring numeric` now records `gsm8k-numeric-v2-last-marker`;
its historical mode remains unchanged. See [the detailed correction and migration
notes](notes/AUDIT_FIXES_2026-10-09.md) for commands, boundaries and tests.
