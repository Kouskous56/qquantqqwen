# 11B1 - SparseGPT 2:4 3B (nsamples 32, wikitext2, upstream repo + port Qwen2)
Validity: exact 2:4 moi row, global sparsity 0.5000.
| 3B | PPL | free/50 | perm/200 | GSM/50 | cons4 |
|---|---|---|---|---|---|
| dense s0 | 8.76 | 24 | 152 | 40 | 29 |
| Wanda s50 unstructured | 12.98 | 19 | 94 | 15 | 7 |
| SparseGPT 2:4 | 15.23 | 10 | 52 | 2 | 0 |
Ket luan: 2:4 co compensation van TE HON unstructured s50 o moi metric.
Compensation khong cuu duoc constraint cung o 3B (voi 32 samples).
Caveat lon: paper dung 128 samples; 32 co the thieu Hessian quality.
Buoc tiep neu lam: nsamples 64/128 (can VRAM) hoac chap nhan 2:4 khong viable o setup nay.
Rot: 19/13/13/7 (r3 sap), dist B72/A51/C52/D25.
