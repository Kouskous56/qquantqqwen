# Quantization, Sparsity, and Behavioral Robustness of Qwen LLMs under Edge Hardware Constraints

This repository accompanies the technical report of the same name
(`paper/paper_v1.pdf`). The central finding is that **similar perplexity
degradation under pruning can correspond to substantially different behavioral
degradation across Qwen2.5 model scales**.

| Scale | ΔPPL@s30 | Free retention |
|---|---|---|
| 0.5B | +7.9% | 65% (V3.2 typed: 11/17) | 71% (V3 unified: 12/17) | 67% (V3 unified: 10/15) |
| 1.5B | +6.5% | 85% (V3.2 typed: 28/33) |
| 3B | +6.3% | 95% (V3.2 typed: 35/37) |

*The &gt;100% value is from a finite 50-item suite and is not interpreted as an
improvement from pruning.

## Repository map

REPRODUCE = reproduce/, MANIFESTS = manifests/, RESULTS = results/, DATA = data/v2/, LEGACY = legacy/, NOTES = notes/, BENCH = bench/, ADAPTERS = third_party_adapters/, PAPER = paper/

reproduce holds canonical runnable entry points; manifests holds frozen provenance; results holds canonical aggregates; data/v2 holds the authored dataset (CC BY 4.0); legacy holds history; notes holds lab notes (see notes README).
## Reproduction
See `REPRODUCIBILITY.md` for the claim → script → manifest → output map.
Model weights are NOT included (72 GB, Qwen license). Download Qwen2.5-0.5B/1.5B/3B-Instruct
from Hugging Face and point the scripts at them. Key commands are listed per claim.

## Citation
See `CITATION.cff`.
