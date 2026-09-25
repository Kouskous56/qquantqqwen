# 10D - 1.5B sparsity scaling (RUN_10D_15B.json)
| 1.5B | PPL | free/50 | perm/200 | GSM/50 | B-rate | cons4 | rot |
|---|---|---|---|---|---|---|---|
| s0 | 8.86 | 38 | 132 | 3 | 0.39 | 18 | 37/38/20/37 |
| s20 | 9.04 | 38 | 129 | 4 | 0.45 | 17 | 37/38/21/33 |
| s30 | 9.39 | 29 | 120 | 3 | 0.32 | 14 | 34/35/24/27 |
## Retention (so 0.5B: free 100%/50%, PPL +1.8%/+7.2%)
- free: s20 100%, s30 76% -> band 75-90%: SCALE GIUP RO
- perm: s20 98%, s30 91%
- PPL delta: s20 +2.0%, s30 +6.0%
## Hypotheses:
- H1 scale robustness: DUNG (76% vs 50%)
- H2 s20 safe: DUNG (100%)
- H3 PPL underestimate: DUNG (PPL +6% ca 2 scale, behavior 76% vs 50%)
- H4 bias giam theo scale: MOT PHAN (cons 18 vs 1-2, nhung r2 van dip, B-rate ~0.4)
## Caveat: GSM 3-4/50 thap bat thuong (thua ca 0.5B-s30) - nghi harness manual
## template khong hop 1.5B. Can rerun voi apply_chat_template + log raw.
