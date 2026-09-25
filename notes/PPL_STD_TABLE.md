# Standardized PPL (512/256, token-weighted, no template, FP16) - RUN_PPL_STD.json
| Scale | s0 | s30 | Δs30 | s40 | s50 | sgpt-2:4 |
|---|---|---|---|---|---|---|
| 0.5B | 15.11 | 16.31 | +7.9% | - | 22.13* | - |
| 1.5B | 9.93 | 10.58 | +6.5% | - | - | - |
| 3B | 8.76 | 9.31 | +6.3% | 10.50 | 12.98 | 15.23 |
(* s50-0.5B so cu, chua rerun chuan - ghi ro)
Tra loi 3 cau hoi:
1. ΔPPL s30 gan nhu nhau (+7.9/+6.5/+6.3%) trong khi behavior 50/76/125% -> mapping phu thuoc scale, MANH HON.
2. Cliff: 9.31 < 10.50 << 12.98, behavior dong thoi down -> sach.
3. Wanda-s50 (12.98) vs SparseGPT-2:4 (15.23) cung evaluator -> 2:4 te hon, confirmed.
PPL cu giu lam historical log; paper chi dung bang nay.
