# REPRODUCIBILITY — claim → command → manifest → expected output
Models (download once from Hugging Face): Qwen2.5-{0.5B,1.5B,3B}-Instruct.
Set MODELROOT to your weights directory. All commands run on CUDA.

## Standardized PPL (Claim: ΔPPL@s30 ≈ +6–8% all scales)
```bash
python reproduce/eval_ppl.py --model $MODELROOT/Qwen2.5-0.5B-FP16 \
  --tok $MODELROOT/Qwen2.5-0.5B-FP16 --corpus bench-corpus/wikitext_test.txt \
  --out results/ppl_05B_s0.json
```
Expected: `05B-s0 ≈ 15.11`, `05B-s30 ≈ 16.31`, `15B-s0 ≈ 9.93`,
`15B-s30 ≈ 10.58`, `3B-s0 ≈ 8.76`, `3B-s30 ≈ 9.31` (see `results/standardized_ppl.csv`).

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
only because the three scales start from different ceilings (18 / 40 / 42), and
0.5B's number is two-sided noise (3 pruned-wrong items became pruned-right).
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
Every decisive run stores versions, dataset hashes and run IDs — see `manifests/`.
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
| `scorer_sha256` matches the script in this repo | — |
| GGUF hash identical to the lab original | — |

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

- 7 of 48 have no resolvable artifact (`gguf_available: false`) — pruned FP16
  directories were deleted during cleanup, so there is nothing left to hash.
- 12 of 48 have no identifiable scoring script
  (`scorer_status: khong-xac-dinh-duoc`, `scorer: null`): 6 `RUN_QWEN3_TYPED_*`,
  5 R1 MMLU (their record shape matches no script on disk), 1 smoke probe.
- `timestamp` is the manifest's file mtime, not the original run time. This is
  stated in `timestamp_source`. File mtimes are not preserved by git, so treat
  these dates as unreliable.

**`dataset_sha256` means different things per schema**, which is why
`dataset_sha256_scope` is mandatory and `detail_schema` is one of four values:

| `detail_schema` | hash covers | can it prove the dataset is unchanged? |
|---|---|---|
| `mmlu` | (i, subject, gold) | yes |
| `gsm` | (i, expected answer) | yes |
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
All Qwen3 scripts accept `--notes-dir` (default `D:/qwen/notes`, the lab path
used for every number in this repo) and `--endpoint` (default
`http://127.0.0.1:11434`). Pass your own values on another machine; the
defaults reproduce the lab layout byte-for-byte, which is why historical
hashes keep matching.

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
