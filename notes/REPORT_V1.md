# Technical Report V1: Quantization, Pruning and Behavioral Robustness of Qwen LLMs for Edge Deployment
*Edge-AI research-engineering study on a single RTX 4050 6GB laptop — 23–24/09/2026.*

## 1. Motivation & hardware constraints
Mục tiêu: chạy Qwen local cho Edge AI bằng quantize + pruning. Phần cứng: Acer Nitro,
RTX 4050 Laptop 6GB, RAM 16GB, Windows 11. Ràng buộc VRAM 6GB loại mọi model >8B FP16,
buộc toàn bộ nghiên cứu xoay quanh 0.5B–4B và câu hỏi: nén bao nhiêu thì còn dùng được.

## 2. Experimental environment
- Ollama 0.34.3 (KEEP_ALIVE=1m, 100% GPU sau khi gỡ kẹt VRAM), 6 models.
- Python 3.11: torch 2.9.0+cu129 (cuda=True), transformers 5.17.0, accelerate 1.15.0,
  datasets 5.0.1, peft 0.21.0, llama-cpp-python 0.3.35 (build từ source).
- Toolchain: cmake 4.4.3, MSVC 14.44, llama.cpp CPU build (llama-quantize, llama-cli),
  Wanda (tự viết lại cho Qwen2: scripts/wanda_qwen.py).
- Dữ liệu: WikiText-2, GSM8K (openai/gsm8k main), bộ 50 toán tự soạn (DE50_MATH.md).
- Kỷ luật từ 9A: frozen harness (greedy, chat template, n_ctx/max_tokens cố định),
  manifest versions + dataset hash + run ID cho mọi run quan trọng.

## 3. Quantization study
- 0.5B: F16 948MB → Q8_0 506 → Q4_K_M 379 → Q3_K_M 339MB. Q3 lặp câu hỏng.
- 3B: F16 5.9GB → Q8 3.1 → Q5 2.1 → Q4 1.8 → Q3 1.5GB. Q4 ngang official, Q3 mất chi tiết.
- 50 toán: mine-Q4 36 vs official 35; V2-perm 149 vs 147/200; agreement 183/200
  cùng chữ, 188/200 cùng đúng/sai; consistency 4/4 cùng 24/50 (chia sẻ 23).
- Kết luận: **Q4_K_M là compression sweet spot**, preserve behavior 91–98%.

## 4. Benchmark failures and V2 methodology
- TEST20: evaluator false-positive (đáp án sai vẫn PASS).
- TEST50 V1: phát hiện B-collapse — 0.5B trả B 47–48/50; đáp án chuẩn lệch B22/C21
  nên raw accuracy 44–50% là ảo; accuracy trên câu non-B chỉ ~10%.
- Nghi pipeline bug: raw prompt (không chat template) → B-rate 1.00, single-char 0/40;
  có template → B-rate 0.28, single-char 40/40. Nhưng rerun full V2 với template:
  collapse CHUYỂN HƯỚNG sang A (Q4 chọn A 155/200) — pathology thật, hướng phụ thuộc format.
- V2 cuối: 50 câu × 4 rotations (đáp án đúng xoay đều A/B/C/D) + free-response,
  log raw đầy đủ. 9A chứng minh variance 0 trên 3 reps.
- Bài học: luôn đo B-rate/distribution/consistency, không tin raw accuracy.

## 5. Sparsity–PPL curves (Wanda, WikiText-32 calibration)
- 0.5B: 13.27 → s20 13.51 → s30 14.23 → s50 22.13 → 2:4 structured 63.92 (hủy).
- 1.5B: 8.86 → 9.04 → 9.39. 3B: 8.76 → 8.91 → 9.31 → s40 10.50 → s50 12.98.
- ΔPPL s30 ~6–7% cả 3 scale; s20 ~+2% cả 3 scale. PPL curves gần như song song theo scale.

## 6. Behavioral divergence (finding trung tâm #1)
- s30 0.5B: PPL +7.2% nhưng free-response 18→6 (−67%).
- WikiLoRA trên s30: PPL 14.23→12.63 nhưng free 9→9 (đứng yên).
- Kết luận: **PPL là necessary-but-insufficient** cho compression của instruction/chat models.

## 7. Masked recovery (9C)
- LoRA rank-8 trên s20: merge thường phá sạch sparsity (0.20→0.00); masked merge
  (reapply mask) giữ 0.20, PPL 12.13, free 17/50 (~18 base).
- Caveat: s20 chưa hỏng trước recovery nên đây là "preserve" chứ chưa phải "recover";
  PPL đẹp một phần do trùng domain train.

## 8. Domain-matched recovery (10C)
- s30 + WikiLoRA (C1): free 9, GSM 3/50 — thất bại.
- s30 + GSM8K LoRA masked (C2): sparsity 0.299 giữ nguyên, GSM-heldout 4→14 (×3.5),
  V2-free 9→11, perm 70/200 (mổ ra vẫn bias nặng: rot 26/22/13/9, cons4 2/50).
- Causal chain C0 4 / C1 3 / C2 14 (cùng model+config, khác data) → H1 domain-mismatch thắng.
- Cost of mask: plain 10 vs masked 14 trên GSM — enforce sparsity không tốn gì.
- Kết luận paper-ready: GSM8K recovery generalizes to unseen samples within the same
  task distribution, but not strongly to the broader suite.

## 9. Scale-dependent pruning robustness (finding trung tâm #2)
| Scale | s20 free ret | s30 free ret | s30 GSM ret | s30 perm ret | ΔPPL s30 |
|---|---|---|---|---|---|
| 0.5B | 100% | 50% | - | ~88% | +7.2% |
| 1.5B | 100% | 76% | 83% | 91% | +6.0% |
| 3B | 112% | 125% | 92.5% | 99% | +6.3% |
- Monotonic confirmed. B-rate giảm theo scale (0.89→0.4→0.26).
- 3B cliff nằm giữa s40 và s50: s50 free 79%, perm 62%, GSM 38%, PPL +48%, D-collapse.
- Phát biểu chuẩn: similar relative perplexity degradation does not imply similar
  downstream robustness across scales.

## 10. Edge deployment
- Bench GPU: 0.5B 280 tok/s/555MB, 1.5B 140/1209MB, 3B 82/2161MB, 4B-thinking 61/3129MB.
- Demo chatbot 8 prompts: 3B mine-Q4 8/8 tốt; 0.5B nhanh 3× nhưng refusal tiếng Việt,
  instruction-following yếu. TTFT steady 0.02–0.05s (warm-up bắt buộc, cold-start 5.2s).
- Winner: **Qwen2.5-3B Q4** (2.2GB, ~80 tok/s). Sweet spots kép: compression=Q4, scale=3B.
- 10A end-to-end (s20→masked→Q4→Ollama): free 18/50 = dense, deploy 481MB GPU.

## 11. Limitations
- Single-run nhiều experiment (9A mới chứng minh deterministic cho V2-free).
- Calibration WikiText-32 thay vì C4-128 của paper; s40/s50-3B dùng 8×128 (deviation đã ghi).
- GSM subsample 50/1319; C2-plain thiếu V2-perm; train/test cùng distribution (recovery, không generalization).
- 2:4 mới test 0.5B; unstructured sparsity chưa thành speedup thực (runtime dense).
- RAM/VRAM laptop giới hạn scale >4B và batch lớn.

## 12. Hardware-aware future work (Phase 11)
- Đo realized benefit: size/VRAM/TTFT/prefill-decode/power/J-token cho dense-Q4 vs s30/s40.
- Chứng minh unstructured sparsity ≈ 0 speedup trên dense kernels → cầu nối sang sparse-aware runtime.
- Thử lại 2:4 trên 3B (hardware-friendly) ± SparseGPT compensation.
- Câu hỏi đỉnh: Capability × Compression × Hardware efficiency (co-design).

---
*Artifacts: D:\qwen (models, bench/*, scripts/*, notes/*, manifests RUN_*.json). Reproduce từ notes + script cùng tên.*
