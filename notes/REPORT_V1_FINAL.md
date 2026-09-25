# Technical Report V1-final: Quantization, Sparsity, and Behavioral Robustness of Qwen LLMs under Edge Hardware Constraints
*Research-engineering study on RTX 4050 6GB laptop, native Windows — 23–24/09/2026. FROZEN.*

## 1. Motivation & constraints
Chạy Qwen local cho Edge AI bằng quantize + pruning, trên RTX 4050 6GB / RAM 16GB.
VRAM 6GB loại mọi model lớn, buộc nghiên cứu xoay quanh 0.5B–4B và câu hỏi:
nén bao nhiêu thì còn dùng được — và sparsity có thành benefit thật không.

## 2. Environment (frozen)
Ollama 0.34.3 (KEEP_ALIVE=1m, GPU) | Python 3.11, torch 2.9.0+cu129, transformers 5.17.0,
accelerate 1.15, datasets 5.0, peft 0.21, llama-cpp-python 0.3.35 |
cmake 4.4.3, MSVC 14.44 | llama.cpp CPU build | Wanda tự viết cho Qwen2 |
upstream SparseGPT + port Qwen2 | llm-compressor (clone, không dùng cho sparse) |
WikiText-2, GSM8K, DE50 tự soạn | frozen harness 9A (greedy, template, manifest).

## 3. Quantization — Contribution 1
Q4_K_M là sweet spot cross-scale: 0.5B giảm 60%, 3B giảm 70% size, behavioral
agreement 91–98% vs official/FP16 (V2-perm 149 vs 147, 183/200 same-letter).
Q3 lặp câu (0.5B) / mất chi tiết (3B).

## 4. Benchmark failures → V2 methodology
TEST20 false-positive → TEST50 B-collapse (0.5B trả B 47/50, đáp án chuẩn lệch B22/C21)
→ pipeline bug (thiếu chat template: B-rate 1.00→0.28) → collapse thật nhưng đổi hướng
sang A → V2 permutation+free-response, frozen variance 0. Quy tắc: không tin raw accuracy.

## 5. Sparsity–PPL curves
0.5B: 13.27→13.51→14.23→22.13 (2:4 naive: 63.92). 1.5B: 8.86→9.04→9.39.
3B: 8.76→8.91→9.31→s40 10.50→s50 12.98. ΔPPL s30 ~6–7% cả 3 scale.

## 6. Behavioral divergence — Contribution 2
s30-0.5B: PPL +7.2% nhưng free 18→6 (−67%). WikiLoRA: PPL 14.23→12.63, free 9→9.
Quan hệ PPL→behavior phụ thuộc scale (ΔPPL giống nhau, robustness 50→76→125%).
PPL là necessary-but-insufficient cho compression của instruction/chat models.

## 7. Masked recovery (supporting)
Merge LoRA thường phá sạch sparsity (0.20→0.00); masked merge giữ 0.20, PPL 12.13,
free ~base. Chưa phải "recover" (s20 chưa hỏng) — xem 10C.

## 8. Domain-matched recovery (supporting)
s30+GSM8K masked LoRA: sparsity 0.299 giữ nguyên, GSM-heldout 4→14 (×3.5),
V2-free 9→11. Causal chain C0 4 / C1 3 / C2 14. Recovery domain-specific ở 0.5B.

## 9. Scale-dependent robustness — Contribution 3 (một phần)
s30 free retention: 50% → 76% → 125%; GSM: -/83%/92.5%. Monotonic confirmed.
3B cliff nằm (40%, 50%]: s50 free 79%, perm 62%, GSM 38%, PPL +48%.

## 10. Edge deployment
3B-Q4: 2.2GB VRAM, ~80 tok/s, TTFT steady ~0.02s (warm-up bắt buộc), demo 8/8.
0.5B: nhanh 3× nhưng refusal TV + instruction yếu. Winner: Qwen2.5-3B Q4.
10A end-to-end (s20→masked→Q4→Ollama): free = dense.

## 11. Realized-benefit failure (11A) + 2:4 failure (11B)
Unstructured s30–s50: VRAM y hệt 5.8GB, decode chậm hơn, J/token tệ hơn — zero benefit.
2:4: backend Windows unavailable (cuSPARSELt) + SparseGPT quality collapse
(PPL 15.23, free 10, GSM 2; ns64 chỉ xuống 14.80 → dừng theo gate).
Kết luận khóa: quality-friendly sparsity != hardware-friendly sparsity.

## 12. Limitations
Single-run nhiều nơi; calibration WikiText-32 (s40/s50-3B: 8×128); PPL đổi stride;
GSM subsample 50; C2-plain thiếu perm; train/test cùng distribution;
2:4 mới ns32/ns64 (chưa 128 — chủ động dừng theo gate); chưa sparse kernel.

## 13. Future work (Phase 12+, ngoài V1)
WSL2/Linux, cuSPARSELt, TensorRT-LLM/vLLM, sparse storage, prefill/decode, power,
C3 mixed-data, s40/s50-1.5B, AWQ/GPTQ/FP8. V1 reproducible trên native Windows RTX4050.

---
*Teams: artifacts + manifests RUN_*.json trong D:\qwen. V1 FREEZE tại đây.*
