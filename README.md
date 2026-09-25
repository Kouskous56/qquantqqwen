# Quantization, Sparsity, and Behavioral Robustness of Qwen LLMs under Edge Hardware Constraints

This repository accompanies the technical report of the same name
(`paper/paper_v1.pdf`). The central finding is that **similar perplexity
degradation under pruning can correspond to substantially different behavioral
degradation across Qwen2.5 model scales**.

| Scale | ΔPPL@s30 | Free retention |
|---|---|---|
| 0.5B | +7.9% | 44% (7/16, same-script pair) |
| 1.5B | +6.5% | 76% |
| 3B | +6.3% | 125%* |

*The &gt;100% value is from a finite 50-item suite and is not interpreted as an
improvement from pruning.

## Repository map
- eproduce/\ — canonical runnable entry points (PPL, V2-GGUF, V2-HF, GSM, Wanda, masked recovery, corpus builder)
- \manifests/\ — frozen provenance (versions, hashes, run IDs)
- esults/\ — canonical aggregate outputs (+ \legacy_v2/\ historical runs)
- \data/v2/\ — authored diagnostic dataset (questions + key; CC BY 4.0)
- \legacy/\ — historical development scripts/results (not authoritative)
- otes/\ — contemporaneous notes; superseded interpretations possible (see otes/README.md\)
- \ench/\, \scripts/\ — retained auxiliary/historical utilities
- \	hird_party_adapters/\, \paper/\ — SparseGPT Qwen port notes; frozen v1 PDF

## Reproduction
See `REPRODUCIBILITY.md` for the claim → script → manifest → output map.
Model weights are NOT included (72 GB, Qwen license). Download Qwen2.5-0.5B/1.5B/3B-Instruct
from Hugging Face and point the scripts at them. Key commands are listed per claim.

## Citation
See `CITATION.cff`.
