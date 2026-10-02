# Qwen3-4B block — what these numbers are, and what they are not

Companion to `manifests_qwen3/`. Read this before quoting any Qwen3 figure.

## The single most important caveat

**Qwen2.5 numbers and Qwen3 numbers in this repo are not comparable.** Qwen2.5
scores come from HF/transformers on FP16 weights. Qwen3 scores come from GGUF
through Ollama. Lora-style masked recovery even ran on a different quant (Q4 vs
FP16) across arms in one early comparison. The blocks are kept apart on purpose;
do not average them, do not put them in one table.

## Cost of 4-bit quantization on this model

Qwen3-4B-Instruct-2507, same 400 GSM8K items, same harness, greedy, CoT. The
only variable is arithmetic precision. The GGUF carries an importance matrix
(`Qwen3-imatrix.dat`), and all three quantized artifacts were built with the
same one — so the sweep varies bit width and nothing else.

| precision | bulk bpw | avg bpw | GSM8K-400 | MMLU-200 | McNemar vs F16 (GSM) |
|---|---|---|---|---|---|
| F16 (lossless) | — | 16.0 | **263/400 = 65.8%** | 152/200 = 76.0% | — |
| Q6_K | 6.5625 | 6.564 | 260/400 = 65.0% | 153/200 = 76.5% | p = 0.728 |
| Q5_K_M | 5.5 | 5.735 | **268/400 = 67.0%** | 149/200 = 74.5% | p = 0.533 |
| Q4_K_M | 4.5 | 4.955 | **247/400 = 61.8%** | 148/200 = 74.0% | **p = 0.0226** |
| Q3_K_M | 3.589 | 4.116 | **220/400 = 55.0%** | 151/200 = 75.5% | **p < 0.0001** |
| Q2_K | 2.688 | 3.272 | **230/400 = 57.5%** | (not run) | **p = 0.0001** |

**The threshold is real, and it is now bracketed on both sides.**
Q3_K_M (3.589 body bits) scores 220/400, below Q4 by 27 items with
McNemar p = 0.0004, in the same direction in both blocks (A: 111 vs 122,
B: 109 vs 125). Against F16 the gap is 43 items, p < 0.0001.

So the damage curve across body precision is: nothing measurable at 5.5
and 6.5625, −4.00pp at 4.5, −10.75pp at 3.589, and −8.25pp at 2.688.
Below the threshold the damage does not keep falling: Q2_K (230/400)
is indistinguishable from Q3_K_M (220/400), McNemar p = 0.275, same
direction in both blocks (A: 112 vs 111; B: 118 vs 109). The threshold
is therefore a cliff edge followed by a floor, at least down to 2.8
body bits. One bound of the pre-registered saturation criterion was
marginally missed (|delta| = 10 vs ≤ 8); the decisive statistic (p)
says indistinguishable, and this is recorded as a judgment call rather
than a silent reclassification.

One caveat on symmetry: Q3_K_M is a *mixed* artifact (2265M params at
Q3_K, 377M lifted to Q4_K), while Q4/Q5/Q6 are single-type bodies. A mixed
body cannot be compared to a pure body as cleanly. But the mixture only
makes Q3 look *better* than a pure Q3_K body would — so Q3 scoring worse
despite 377M Q4-strength tensors strengthens, not weakens, the conclusion.

**The result is a threshold, not a curve.** F16, Q6_K and Q5_K_M are mutually
indistinguishable; only Q4_K_M separates. Q5_K_M scores *above* the lossless
control (268 vs 263, p = 0.53), and Q5-vs-Q4 is the strongest contrast in the
whole sweep: 36 vs 15 discordant, p = 0.0046. Same pattern in both 200-question
blocks — Q5 gets 135 and 133 against F16's 133 and 130.

## The variable is not average bpw

Average bpw (4.955 / 5.735 / 6.564) looks like a continuous sweep. It is not.
Reading tensor types shows the three artifacts differ in exactly one thing: the
type of the 180 body tensors.

| artifact | 180 body tensors | body bpw | `token_embd` |
|---|---|---|---|
| Q2_K | Q2_K × 144 + Q3_K × 36 (mixed) | 2.688 | Q6_K |
| Q3_K_M | Q3_K × 144 + Q4_K × 36 (mixed) | 3.589 | Q6_K |
| Q4_K_M | Q4_K × 180 | 4.5 | Q6_K |
| Q5_K_M | Q5_K × 180 | 5.5 | Q6_K |
| Q6_K | Q6_K × 180 | 6.5625 | Q6_K |

`token_embd` (389M params) is Q6_K in all three — the highest-precision tensor in
the model even inside Q4_K_M, which refutes the hypothesis that 4-bit
specifically crushes the embedding. And no artifact exists between 4.955 and
5.735 average bpw, so any mechanism explained in terms of average bits is
explaining a quantity that was never varied.

**Damage appears only at 4.5 bits of body; one extra bit of body recovers it
completely, measurably indistinguishable from 16-bit.**

## Four things this block does NOT show

**1. "Quantization hurts reasoning more than knowledge" is not established.**
The ratio is 2.0×, and GSM reaches significance while MMLU does not. The
defensible statement is narrower: *quantization measurably damaged reasoning;
we did not detect damage to knowledge, but at n=200 we could not have detected
2pp.* Those are different claims and were previously conflated. A second
reading is available and better supported: MMLU in forced-single-letter format
is simply insensitive, so the asymmetry may be a property of the instrument.

**2. The threshold is bracketed, not located.** (3.589, 4.5) body bits. Nothing
in this sweep measures inside that interval. Do not quote 4.5 as the threshold.

**3. Q6-vs-Q4 gives p = 0.079 and that is a power limit, not a finding.**
Q6 is only 3 items below F16 on 400 — a smaller deficit than 400 items can
resolve. Reporting "Q6 matches Q4" would be wrong; so would "Q6 differs from
Q4". The honest line is: *no damage detected at Q6, and this comparison cannot
separate them.*

**4. MMLU-200 has almost no power here.** Across all quantized levels only 24
of 200 items changed answer (17 across the first three levels, 24 with Q3
included). Q6_K scored *above* F16 (153 vs 152), which is
pure noise. Q3_K_M scores 151/200, indistinguishable from F16 (p = 1.0):
even below the damage threshold, multiple-choice knowledge does not move.

## A 200-question block is not a stable unit

Both Qwen3 MMLU and GSM were run as two blocks of 200. The same Q4-vs-F16 GSM
comparison:

| block | Δ | McNemar p |
|---|---|---|
| A (items 0–199) | −5.5pp | 0.0192 |
| B (items 200–399) | −2.5pp | 0.4244 |
| **pooled (400)** | **−4.00pp** | **0.0226** |

Identical models, identical protocol, identical harness. Block A looks
significant, block B does not separate at all. An earlier single-50-question
measurement of the same contrast gave −10pp — inflated 2.5×.

So: the effect is real, but **the magnitude is not pinned down**, and no amount
of re-analysis of the existing data will pin it down. Narrowing it to ±1pp
needs roughly n=1000, about 20 GPU-hours. The honest number to quote is
"roughly 2–5pp" with the pooled point estimate at −4.00pp, not a single crisp
figure.

This block-flip happened twice in one session. It is the most useful thing in
this file.

## Pruning arms (Q4_K_M + imatrix, all five arms)

All arms are quantized with the same recipe, so these compare pruning against
pruning. GSM8K-400: dense 247, s20-c4 250, s20-mixed 253, s30-mixed 251, EoRA
257. **No arm separates from dense** (McNemar p ≥ 0.143 throughout).

The direction reverses versus the project's earlier 4-bit-without-imatrix runs.
That reversal is not a pruning effect — it is an artifact: three arms that had
no importance matrix gained 6/12/7 items when one was added, while arms already
built with a matrix gained 0. The dense↔s20-c4 gap shrank from 12 items to 3.

## What was falsified along the way

- **Embedding crushed to 4-bit** (would explain the GSM drop). Refuted by
  reading tensor types: `token_embd` is Q6_K in Q4_K_M.
- **Quantization damage is a sub-block range artifact.** Refuted by direct
  measurement: per-32-element `max−min` shifts 0.005% on average (max 0.048%)
  between dense and 20%-pruned, versus exactly 0.000% vs 20.000% between dense
  and the pruned model.

## Provenance

41 of 48 manifests resolve to a GGUF still on disk, hashed. 10 come from the
manifest's own record; 18 were resolved by walking
model → Ollama `FROM` blob → SHA-256 → file; 13 R1 manifests resolve
through a verified legacy model-to-file map (blob SHAs matched before the
Ollama models were removed, hashes recomputed from the files on every
run). All 10 hashes that could be
compared against a lab original matched exactly; 0 mismatches.

7 have no resolvable artifact and 12 have no identifiable scorer. These are
recorded as unverifiable rather than filled in. See `REPRODUCIBILITY.md` for
what CI does and does not check.
