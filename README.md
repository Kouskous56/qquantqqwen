# Quantization, Sparsity, and Behavioral Robustness of Qwen LLMs under Edge Hardware Constraints

![repro](https://github.com/Kouskous56/qquantqqwen/actions/workflows/repro.yml/badge.svg)

This repository accompanies the frozen V1 technical report of the same name
(paper/paper1042.pdf). The central finding is that **similar perplexity
degradation under pruning can correspond to substantially different behavioral
degradation across Qwen2.5 model scales**.

| Scale | ΔPPL@s30 | Behavior items lost @ s30 (V3.3 typed) |
|---|---|---|
| 0.5B | +7.9% | 18 → 11 = **−7 items** (61% of 18) |
| 1.5B | +6.5% | 40 → 33 = **−7 items** (82.5% of 40) |
| 3B | +6.3% | 42 → 40 = **−2 items** (95% of 42) |

*V3.3 canonical: typed rescore over frozen V3.1 raw generations; McNemar exact in manifests/RUN_V33_RESCORE.json.*

**Read the absolute counts, not the percentages.** Retention 61% / 82.5% / 95%
suggests "larger models hold up better", but 0.5B and 1.5B lose the *same*
number of items (7) — the percentage difference is a denominator effect, because
0.5B starts from a much smaller ceiling (18 vs 40 items correct). In absolute
points the trend is 7 / 7 / 2, and the significant step is 0.5B→1.5B
(Fisher p = 0.0053); 1.5B↔3B is not significant (p = 0.189). Over the three
scales, Cochran–Armitage χ² = 17.6, p = 0.00003. The 0.5B figure additionally
moves in both directions: 3 items that dense answered *wrong* were answered
*right* after pruning, so its retention number is two-sided noise rather than a
clean degradation signal. See `notes/SCALE_STATS.md` for the
full 2×2 tables.

## Repository map

REPRODUCE = reproduce/, MANIFESTS = manifests/ (Qwen2.5) + manifests_qwen3/ (Qwen3-4B), RESULTS = results/, DATA = data/v2/, LEGACY = legacy/, NOTES = notes/, BENCH = bench/, ADAPTERS = third_party_adapters/, PAPER = paper/

reproduce holds canonical runnable entry points; manifests holds frozen provenance; results holds canonical aggregates; data/v2 holds the authored dataset (CC BY 4.0); legacy holds history; notes holds lab notes (see notes README).

**`manifests_qwen3/` is a separate block and its numbers do not mix with the
Qwen2.5 ones.** It was scored on GGUF through Ollama rather than FP16 through
transformers, on one 6 GB consumer GPU. Those runs cannot be regenerated in CI,
so CI checks their integrity (hashes recomputed from the stored per-item
records) rather than regenerating them — every manifest says `regenerable:
false`. Start with `notes/QWEN3_BLOCK.md`; it also records two hypotheses that
were measured and refuted.

## Reproduction
See `REPRODUCIBILITY.md` for the claim → script → manifest → output map.
Model weights are NOT included. Only the about 10 GB upstream base checkpoints are needed (see reproduce/download_weights.py); the 72 GB figure was the full local workspace with derived artifacts. Download Qwen2.5-0.5B/1.5B/3B-Instruct
from Hugging Face and point the scripts at them. Key commands are listed per claim.

## Citation
See `CITATION.cff`.
