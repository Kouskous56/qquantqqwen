# Quantization, Sparsity, and Behavioral Robustness of Qwen LLMs under Edge Hardware Constraints

This repository accompanies the technical report of the same name
(`paper/paper_v1.pdf`). The central finding is that **similar perplexity
degradation under pruning can correspond to substantially different behavioral
degradation across Qwen2.5 model scales**.

| Scale | ΔPPL@s30 | Free retention |
|---|---|---|
| 0.5B | +7.9% | 50% |
| 1.5B | +6.5% | 76% |
| 3B | +6.3% | 125%* |

*The &gt;100% value is from a finite 50-item suite and is not interpreted as an
improvement from pruning.

## Repository map
- `bench/` — evaluation harnesses (V2 permutation + free-response, GSM8K, PPL, IFEval runners)
- `scripts/` — Wanda pruning port for Qwen2, LoRA recovery, conversion helpers
- `data/v2/` — the 50-question diagnostic suite we authored (CC BY 4.0)
- `manifests/` — run manifests (versions, dataset hashes, run IDs)
- `results/` — aggregate CSVs behind the paper tables
- `notes/` — phase reports (full experimental narrative)
- `third_party_adapters/` — our Qwen2 port of upstream SparseGPT (see README there)
- `paper/` — frozen v1 PDF (LaTeX sources retained by the authors; build requires MiKTeX/XeLaTeX)

## Reproduction
See `REPRODUCIBILITY.md` for the claim → script → manifest → output map.
Model weights are NOT included (72 GB, Qwen license). Download Qwen2.5-0.5B/1.5B/3B-Instruct
from Hugging Face and point the scripts at them. Key commands are listed per claim.

## Citation
See `CITATION.cff`.
