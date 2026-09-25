# Báo cáo đầy đủ Phase 11B: Structured 2:4 sparsity — từ backend tới quality

## 1. Động cơ
11A chứng minh unstructured sparsity không tạo realized benefit trên dense runtime.
2:4 structured là ứng viên duy nhất hardware hiểu được (Sparse Tensor Cores).
Câu hỏi kép: (a) backend hiện tại có chạy được 2:4 không? (b) 2:4 có giữ capability không?

## 2. 11B0 — Backend feasibility (5 phút, phủ định nhanh)
- torch 2.9 sparse API có, GPU arch 8.9 đủ điều kiện.
- `to_sparse_semi_structured` fail cả 4 shapes Qwen-3B: cuSPARSELt not supported
  (PyTorch pip Windows không kèm backend).
- Kết luận: semi-structured path của PyTorch không khả thi native Windows.
  Không đốt thêm giờ debug — đúng mục đích feasibility gate.

## 3. 11B1 — SparseGPT 2:4 3B (quality feasibility)
- Tooling: upstream IST-DASLab/sparsegpt (KHÔNG dùng llm-compressor latest vì
  sparse đã deprecated) + port Qwen2 (Qwen2ForCausalLM, AutoTokenizer,
  position_embeddings tự tính cho transformers 5.x, fix datautils Salesforce/wikitext).
- Config: nsamples 32, wikitext2, prunen 2 prunem 4, percdamp 0.01, blocksize 128.
- Validity: exact 2:4 mọi row-group, global sparsity 0.5000.

## 4. Kết quả
| 3B | PPL | free/50 | perm/200 | GSM/50 | cons4 |
|---|---|---|---|---|---|
| dense s0 | 8.76 | 24 | 152 | 40 | 29 |
| Wanda s50 unstructured | 12.98 | 19 | 94 | 15 | 7 |
| SparseGPT 2:4 | 15.23 | 10 | 52 | 2 | 0 |
- 2:4 + Hessian compensation vẫn TỆ HƠN unstructured s50 mọi metric.
- Rot 19/13/13/7 (r3 sập), B-rate 0.36.
- Trả lời: compensation KHÔNG làm 2:4 usable ở 3B (với 32 samples).

## 5. Caveats (ghi thẳng)
- Paper dùng nsamples 128; ta dùng 32 (VRAM 6GB). Hessian quality có thể thiếu —
  đây là confound lớn nhất. Muốn công bằng: thử 64/128.
- Naive Wanda-2:4 chỉ test ở 0.5B (PPL 63.92), không phải baseline cross-scale công bằng.
- Chưa có sparse-kernel measurement (11B0 đã chặn) nên đây thuần quality study.

## 6. Kết luận và hướng (KHÓA)
Phase 11B exposed a hardware–quality mismatch. Unstructured Wanda pruning preserved
Qwen2.5-3B behavior relatively well through roughly 40% sparsity, but produced no
realized acceleration under dense execution. Conversely, hardware-compatible 2:4
sparsity was unavailable through the native Windows PyTorch backend and, when produced
with SparseGPT, caused severe degradation. Doubling SparseGPT calibration from 32 to 64
samples reduced PPL only from 15.23 to 14.80 (2.8%), below the predefined continuation
threshold. We therefore stop the 2:4 branch rather than escalating calibration or backend setup.

Trong phạm vi 32→64 đã thử, calibration size không phải limitation chính;
rigid 2:4 constraint vẫn là giải thích hàng đầu. (Không claim về 128 vì chủ động không chạy.)

$$
quality-friendly sparsity != hardware-friendly sparsity
$$

Đây không chỉ là "2:4 thất bại" — là kết quả nối model compression với hardware constraints.

## 7. Files
tools/sparsegpt/qwen.py (+ datautils patch) | bench/bench_11b0.py | bench/eval_11b1.py |
notes/BENCH11B0.txt, RESULT_11B1.md, RUN_11B1.json | models/Qwen2.5-3B-sparsegpt-24.
