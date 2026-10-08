# Math50 V4 (4.0.0)

**Superseded: do not use for new experiments.** Symbolic normalization can
incorrectly accept `x^23` for the derivative of `x^3`. Use
[4.0.1](../v4_0_1/README.md). Frozen sources and historical migration below
remain unchanged for audit.


This is a separately versioned, authored **mathematics diagnostic**, not a general
knowledge benchmark. The 50 item IDs are retained for traceability. All 50 keys,
distractors and answer types were reviewed; 18 stems were clarified or corrected.
See [the complete item review](ITEM_REVIEW.md). V2, original manifests, V3.3
aggregates and the frozen paper remain historical evidence.

## What changed

- `o38` now requests **both** possible units digits, 4 and 6. The former
  choice-dependent question could not justify accepting only 4 in free response.
- `h29` specifies the doubling rule rather than inferring a unique sequence.
- `a44` explicitly requires algebraic multiplicity: `1,1`, not a deduplicated set.
- Domains, radians, units, variable of differentiation and integration constants
  are explicit where necessary. All free-response instructions are newly versioned.
- Numeric equality uses exact rational arithmetic; there is no last-number
  extraction. Ordered pairs preserve order; multisets preserve multiplicity.
- Symbolic scoring uses published finite aliases, **not** general algebraic
  equivalence. Equivalent expressions absent from the alias list may fail.

## Frozen experiment protocol

`protocol.json` is authoritative. Free mode requests 50 answers with at most 64
generated tokens each. MCQ mode requests 200 answers with at most 8 tokens each:
each question appears under all four cyclic choice rotations, in separate requests.
Each answer letter is correct exactly 50 times. Report both response accuracy /200
and all-four-rotations-correct /50. These are **50 clustered questions**, not 200
independent samples. There is one greedy pass, seed 0, temperature 0, context 2048,
repeat penalty 1, thinking disabled. These budgets are prospective protocol choices,
not evidence that truncation has been eliminated; report `length` terminations.

The supported executable backend is local Ollama `/api/chat`, with a separate
request for every question. Its manifest digest, exact GGUF SHA256, template,
system prompt, parameters, server version and host are recorded. `/api/show` must
bind `FROM` to that GGUF; adapter layers are rejected. The GGUF contains the
tokenizer. A hash of the complete show response records remaining model metadata.
Each request prompt, full prompt plan, generation policy, dataset, protocol and
scorer have content identities. Ollama does not expose rendered input token IDs
through this API; the request text plus template/system hashes are the reproducible
boundary. This runner does not claim token-identical equivalence with Transformers.

Model identity is rechecked after every response; interrupted runs save an atomic
checkpoint and require explicit `--resume`. Resume requires identical identities,
ordered prompt hashes, valid telemetry and scores recomputed from stored raw text.
Do not edit a completed manifest to introduce new generations under an old run ID.
The ID identifies a configuration, not an independent statistical replicate.

## Run from the repository root

Requires Python 3.11+; the runner itself uses only the standard library.
Ollama must already serve the desired local GGUF model.

```bash
# Inspect every exact prompt without loading or querying a model.
python -m reproduce.eval_suite_v4 --mode free --prepare-only --out runs/v4/free-prompts.json
python -m reproduce.eval_suite_v4 --mode mcq --prepare-only --out runs/v4/mcq-prompts.json

# Replace model name and GGUF path with the same local artifact.
python -m reproduce.eval_suite_v4 --mode free --model MODEL_NAME --gguf /path/model.gguf --out runs/v4/free.json
python -m reproduce.eval_suite_v4 --mode mcq --model MODEL_NAME --gguf /path/model.gguf --out runs/v4/mcq.json
# To resume, repeat exactly the same command and append --resume.

# Audit public historical outputs without generating new answers.
python -m reproduce.rescore_suite_v4 --out-dir runs/v4/migration-audit
python -m pytest tests/test_suite_v4.py -q
```

Outputs must use new paths. Historical source/evidence directories are protected
by the runner. The release lock is checked before prompt export, evaluation and
rescoring; modified source or dataset bytes fail closed. Line endings are pinned
to LF. A future change to any frozen component requires a new release/version and
new migration evidence; do not silently refresh this release's lock.

## Scoring contract and limits

Whole-response dollar/LaTeX/boxed wrappers and terminal transport tokens are allowed.
An anchored `Answer:`, `Final answer:` or `####` is accepted for correctness but
flagged as noncompliant format. Other prose, units and variable assignments are
not extracted. MCQ permits exactly one letter (case insensitive) after these same
wrapper rules. Numeric fractions, decimals, bounded scientific notation and scalar
thousands separators are allowed; comma in a multiset or tuple is a delimiter.
Blank answers are `empty`; invalid parsings are `unparseable`; valid wrong answers
are `incorrect`. All count as failures in the fixed denominator.

`format_compliant` means a nonempty parseable response without an answer prefix;
it is independent of correctness and is not a general grammar or prose detector.
Symbolic expressions use a restricted token vocabulary and finite aliases; text
labels normalize case, whitespace and final periods. This is a deterministic
diagnostic scorer, not an arbitrary mathematical proof checker.

Compare runs only within identical protocol, prompt plan, scorer and backend
policy, explicitly declaring varied model artifacts. Report paired per-item gains
and losses, per-band scores, parse failures and truncation. The old easy/hard band
names are inherited labels, not empirically calibrated difficulty levels.
Repeated greedy outputs are stability checks, not extra independent observations.

The [migration audit](../../results/v4_migration/MIGRATION.md) separates stored
labels, recomputed V3 scores, V4 rules on **old answer keys**, and the 32 unchanged
stems with new aliases. None is a fresh V4 benchmark: all original prompts and
generation budgets were historical. Qwen3's original scorer provenance remains
unknown. Fresh comparative GPU runs belong to a subsequent experimental task.

API references: [Ollama chat](https://docs.ollama.com/api/chat),
[local model listing](https://docs.ollama.com/api/tags).
Dataset license: [CC BY 4.0](LICENSE); author attribution follows the repository
[citation metadata](../../CITATION.cff).
