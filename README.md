# Quantization, Sparsity, and Behavioral Robustness of Qwen LLMs under Edge Hardware Constraints

![repro](https://github.com/Kouskous56/qquantqqwen/actions/workflows/repro.yml/badge.svg)

This repository accompanies the technical report of the same name
(manuscript temporarily omitted while the reproducibility package is finalized;
the arXiv version will be linked at release). The central finding is that **similar perplexity
degradation under pruning can correspond to substantially different behavioral
degradation across Qwen2.5 model scales**.

| Scale | ΔPPL@s30 | Free retention |
|---|---|---|
| 0.5B | +7.9% | 61% (V3.3 typed: 11/18) |
| 1.5B | +6.5% | 82.5% (V3.3 typed: 33/40) |
| 3B | +6.3% | 95% (V3.3 typed: 40/42) |

*V3.3 canonical: typed rescore over frozen V3.1 raw generations; McNemar exact in manifests/RUN_V33_RESCORE.json.*

## Repository map

REPRODUCE = reproduce/, MANIFESTS = manifests/, RESULTS = results/, DATA = data/v2/, LEGACY = legacy/, NOTES = notes/, BENCH = bench/, ADAPTERS = third_party_adapters/

reproduce holds canonical runnable entry points; manifests holds frozen provenance; results holds canonical aggregates; data/v2 holds the authored dataset (CC BY 4.0); legacy holds history; notes holds lab notes (see notes README).
## Reproduction
See `REPRODUCIBILITY.md` for the claim → script → manifest → output map.
Model weights are NOT included. Only the about 10 GB upstream base checkpoints are needed (see reproduce/download_weights.py); the 72 GB figure was the full local workspace with derived artifacts. Download Qwen2.5-0.5B/1.5B/3B-Instruct
from Hugging Face and point the scripts at them. Key commands are listed per claim.

## Citation
See `CITATION.cff`.
