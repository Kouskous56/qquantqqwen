# 10F - 3B sparsity cliff (s0/s20/s30/s40/s50) - HOAN CHINH
| 3B | PPL | free/50 | perm/200 | GSM/50 | cons4 |
|---|---|---|---|---|---|
| s0 | 8.76 | 24 | 152 | 40 | 29 |
| s20 | 8.91 | 27 | 155 | 41 | 31 |
| s30 | 9.31 | 30 | 151 | 37 | 29 |
| s40 | 10.50 | 26 | 138 | 31 | 25 |
| s50 | 12.98 | 19 | 94 | 15 | 7 |
Retention vs s0: free 112/125/108/79%, perm 102/99/91/62%, GSM 102/92/78/38%.
PPL delta: +1.7/+6.3/+19.9/+48.2%.
CLIFF giua s40 va s50: free rot manh, perm sap 91->62%, GSM 78->38%, D-collapse (D82), cons 25->7.
Ket luan: safe regime 3B ket thuc o ~40%. Cliff thuc nam (40%, 50%].
Deviations: s40/s50 calibration 8x128 (s20/s30: 16x256); PPL deu stride256-mx512 tu s40 (s0-s30: stride512-mx1024) - ghi de tai lap.
