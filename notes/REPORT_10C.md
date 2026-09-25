# Báo cáo chi tiết 10C: Domain-matched recovery trên s30

## 1. Bối cảnh và câu hỏi
10B để lại 2 giả thuyết cạnh tranh cho thất bại của WikiText-LoRA trên s30:
- (H1) domain mismatch — sai data, sai objective;
- (H2) s30 damage vượt khả năng cứu của LoRA rank-8/300 samples.
10C phân biệt bằng recovery matched-domain (GSM8K), giữ nguyên mọi thứ khác.

## 2. Thiết kế thí nghiệm
- Arms: C0 = s30 không recovery | C1 = s30 + WikiLoRA (từ 10B) | C2 = s30 + GSM8K LoRA (bản masked giữ sparsity + bản plain tham khảo).
- Config LoRA giống hệt 10B: rank 8, alpha 16, dropout 0.05, lr 2e-4, AdamW,
  150 steps (300 samples, batch 2), target toàn bộ attention + MLP projections.
- Khác duy nhất ở C2: loss chỉ trên assistant tokens, format chat template,
  train trên GSM8K-train (TUYỆT ĐỐI không chạm test). Loss train 0.47 → 0.44.
- Mask: s30-masks.pt (168 tensors), reapply sau merge.
- 4 trục đo: sparsity | V2 free-response/50 | V2 permutation/200 | GSM8K-heldout/50 (test split).
- Threshold thành công đặt TRƯỚC: V2 free 9 → 14+, giữ sparsity ~0.30.

## 3. Kết quả
| Arm | sparsity | V2 free | V2 perm | GSM-heldout/50 |
|---|---|---|---|---|
| C0 s30 base | 0.299 | 9 | - | 4 |
| C1 s30+WikiLoRA | 0.299 | 9 | - | 3 |
| C2 plain (mất sparsity) | 0.000 | 11 | - | 10 |
| C2 masked | 0.299 | 11 | 70 | 14 |

Mổ permutation của C2-masked: rotations 26/22/13/9, k-dist 1/4 chiếm 34/50,
cons4 2/50, phân bố B105/A74/C12/D9 (B-rate 0.525).

## 4. Diễn giải
a) Threshold generic (free 9→14+) KHÔNG đạt (11). Không đổi tiêu chuẩn sau khi thấy số.
b) GSM-heldout 4 → 14 (×3.5, +20 điểm %) ĐẠT VƯỢT, với sparsity nguyên vẹn.
   Đây là evidence mạnh cho H1 trong domain toán.
c) Causal chain khóa đẹp: C0 4, C1 3, C2 14 — cùng model, cùng config LoRA,
   chỉ khác data → recovery khác hẳn. WikiLoRA gần như bằng không trên toán.
d) Cost of enforcing sparsity trên chính task recovery: plain 10 vs masked 14.
   Enforce mask KHÔNG tốn gì, thậm chí như một dạng regularization.
e) Perm 70 cần đọc thận trọng: rotation r0/r1 cao (26/22), r2/r3 thấp (13/9),
   kdist vẫn dồn ở 1/4 (34/50), cons4 chỉ 2/50. Cải thiện một phần là thật
   (hơn dense-F16 64/200), nhưng bias vị trí còn nặng — không gọi là reasoning recovery.
f) PPL không được dùng làm criterion thành công (bài học Task 8).

## 5. Kết luận
Domain-matched masked adaptation phục hồi same-domain generation gấp 3.5 lần
trong khi giữ 29.9% sparsity. Behavioral recovery là CÓ THỂ nhưng
domain-specific ở quy mô 0.5B: V2-free generic chỉ 9→11.
Công thức paper-ready: GSM8K recovery generalizes to unseen samples within the
same task distribution, but does not generalize strongly to the broader suite.

## 6. Limitations (ghi thẳng)
- GSM-heldout 50 câu subsample (đợi 1319 câu đầy đủ khi cần nghiêm túc).
- Single run; 9A mới chứng minh deterministic cho V2-free, GSM-generate chưa test repeat.
- Train/test GSM8K cùng distribution: kết luận recovery, không kết luận generalization.
- 0.5B capacity ceiling có thể chặn generic recovery bất kể data.
- C2-plain thiếu V2-perm (chỉ đo free + GSM); so mask-cost mới đủ 2/4 trục.

## 7. Files và reproducibility
- scripts/recover_gsm8k.py | bench/eval_10c.py | bench/eval_10c_patch.py
- notes/DESIGN_10C.md | notes/RESULT_10C.md | REPORT này
- models/s30-gsm8k-adapter | models/s30-gsm8k-masked | models/s30-masks.pt
- Baseline: TEST50_V2_CHAT.json | dataset openai/gsm8k main | harness frozen 9A
  (n_ctx 1024, max_tokens 30/5, temperature 0, greedy, chat template).
