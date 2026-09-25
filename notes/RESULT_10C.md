# 10C kết quả (s30 + GSM8K LoRA, masked, sparsity 0.299 giữ nguyên)
| Arm | sparsity | V2 free | V2 perm | GSM-heldout/50 |
|---|---|---|---|---|
| C0 s30 base | 0.299 | 9 | - | 4 |
| C1 s30+WikiLoRA | 0.299 | 9 | - | 3 |
| C2 plain (mất sparsity) | 0.000 | 11 | - | 10 |
| C2 masked | 0.299 | 11 | 70 | 14 |
Nhận xét:
- Threshold free 9→14+ KHÔNG đạt (11). Nhưng GSM-heldout 4→14 (×3.5) ĐẠT vượt.
- Causal chain khóa: C0 4, C1 3, C2 14 — cùng model+config, khác data → khác recovery.
- Cost of mask trên task recovery: plain 10 vs masked 14 — enforce sparsity KHÔNG tốn, thậm chí như regularizer.
- Perm 70 mổ ra: rot 26/22/13/9, kdist 1/4 chiếm 34/50, cons4 2/50, B-rate 0.525 — vẫn bias nặng, cải thiện một phần thật.
- Recovery là domain-specific: matched data cứu same-domain generation, không lan sang generic V2-free.
