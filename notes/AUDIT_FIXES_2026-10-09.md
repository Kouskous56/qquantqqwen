# Three evaluation audit fixes

## Math50 symbolic scoring: release 4.0.1

4.0.0 removed multiplication signs between operands, collapsing the correct
alias `x^2*3` into the wrong answer `x^23`. The new scorer tokenizes expressions
and retains operators. Implicit multiplication is made explicit, so `3x^2`,
`3*x^2` and `3\cdot x^{2}` share a representation without conflating exponent
digits with a multiplier. This remains a finite-alias scorer, not a CAS.

Use `python -m reproduce.eval_suite_v401` and `data/v4_0_1/`. All four original
V4 modules, the 4.0.0 release lock, questions, protocol and migration results
remain frozen. The 4.0.1 release has independent versions, source hashes and
migration outputs. Its 50 mathematical questions and all 250 prompts are
unchanged. The historical 1,200-response migration has unchanged aggregate
scores; the synthetic regression `x^23` now correctly fails.

Do not resume a 4.0.0 checkpoint with 4.0.1. The release identity check rejects
it. Use a new output path. Old release documentation is marked superseded.

## Qwen3 checkpoint artifact identity

Shared GSM/MMLU runners now use identity version 2. Before loading a checkpoint,
the runner resolves the exact model tag through `/api/tags`, reads `/api/show`,
and checks the manifest digest again to detect a concurrent alias change.
It extracts the GGUF digest from `FROM` and hashes the actual local GGUF bytes.
A model-manifest digest is not treated as a weight-file digest.

Identity includes the GGUF hash, manifest hash, artifact size/mtime, template,
system prompt, parameters, model metadata, capabilities and Ollama version.
The configuration fields are covered by a canonical SHA256. Dataset, source
and generation-token-budget identities remain included. A declared round-two
GGUF hash must agree with the served artifact.

Before and after every response, metadata/configuration and file stat must
still match. No response observed across an identity change is checkpointed.
At completion the whole GGUF is hashed again. A resumed run also hashes it
before accepting any prior responses. Existing version-1 checkpoints lack
this evidence and are rejected rather than silently upgraded.

Local servers may use the blob path returned by `FROM`. All wrappers now accept
`--gguf`; it is required for remote servers or when the server's blob path is
not accessible to the client. Example:

```bash
python bench/qwen3/gsm_sweep.py --model MODEL_TAG --tag NEW_RUN --precision Q4_K_M \
  --gguf /path/exact-model.gguf --notes-dir runs/gsm --n 200 --offset 0
```

Adapter layers are rejected by this GGUF-only policy. Identity checks assume
the server honestly reports its manifest and loads content-addressed blobs;
they are not cryptographic attestation of a remote server or proof against an
adversarial alias swap that is reverted between observations. Keep model tags
unchanged during an experiment.

## GSM numeric protocol v2

`--scoring numeric` now records `gsm8k-numeric-v2-last-marker`. The parser selects
the rightmost `\boxed{` or `####` by position, including an invalid or unclosed
final marker. It never falls back to an earlier answer when the explicit final
answer is invalid. Malformed decimal/fraction tokens such as `3.4.5`, `1,23`,
and `1/2/3` are rejected. A sentence-final period is allowed. Numeric lengths
and exponents are bounded before conversion to exact rational numbers.

Examples: `\boxed{2} Correction: #### 3` becomes 3; `#### 2 \boxed{3}` also
becomes 3; `\boxed{2} #### unknown` is invalid. `--scoring historical` and its
extractor retain the original semantics and protocol label.

## Verification scope

Tests cover all declared aliases and distractors, wrong exponent collisions,
mixed marker order, malformed final answers, artifact-byte mismatches, same-tag
retargeting, template changes, changes during a response and legacy checkpoints.
A real loopback HTTP integration test exercises actual requests, actual file
hashing, interruption/resume and alias retargeting against a synthetic inference
server. It is explicitly not a language-model benchmark. GPU smoke evidence,
when available, is kept separately from these synthetic tests.
