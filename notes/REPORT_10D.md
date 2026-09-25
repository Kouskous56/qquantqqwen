# Báo cáo đầy đủ 10D: Scale-dependent sparsity tolerance (0.5B → 1.5B)

## 1. Câu hỏi trung tâm
Khi tăng 0.5B → 1.5B, ngưỡng chịu pruning có dịch lên không? Preregister 4 hypotheses
(H1 scale robustness, H2 s20 safe, H3 PPL-underestimate, H4 bias giảm theo scale).

## 2. Thiết kế (controlled scaling)
- Wanda `scripts/wanda_qwen.py`, calibration WikiText-32, sparsity định nghĩa giống 0.5B.
- KHÔNG cải thiện calibration cho 1.5B (dù C4-128 có thể tốt hơn) — model size là biến duy nhất.
- Matrix 1.5B s0/s20/s30 (chưa s50, chưa Q4, chưa LoRA): PPL + V2 free/50 + perm/200
  (full dissection: rotations, distribution, cons4) + GSM-heldout/50.
- Frozen harness 9A: chat template, greedy, n_ctx 1024.

## 3. Kết quả
| 1.5B | PPL | free/50 | perm/200 | GSM/50 (fix) | B-rate | cons4 | rot r0-r3 |
|---|---|---|---|---|---|---|---|
| s0 | 8.86 | 38 | 132 | 35 | 0.39 | 18 | 37/38/20/37 |
| s20 | 9.04 | 38 | 129 | 32 | 0.45 | 17 | 37/38/21/33 |
| s30 | 9.39 | 29 | 120 | 29 | 0.32 | 14 | 34/35/24/27 |

## 4. Retention (metric so scale công bằng)
| | 0.5B | 1.5B |
|---|---|---|
| free s20 | 100% (18/18) | 100% (38/38) |
| free s30 | 50% (9/18) | 76% (29/38) |
| perm s30 | ~88% (56/64) | 91% (120/132) |
| ΔPPL s20 | +1.8% | +2.0% |
| ΔPPL s30 | +7.2% | +6.0% |

## 5. Đối chiếu hypotheses
- H1: ĐÚNG. s30 free retention 76% vs 50% — band 75–90% "scale giúp rõ".
- H2: ĐÚNG. s20 giữ 100% free, 98% perm.
- H3: ĐÚNG KIỂU MỚI. ΔPPL s30 gần như nhau hai scale (+6.0% vs +7.2%)
  nhưng behavior 76% vs 50% — cùng LM-loss degradation, robustness khác theo scale.
- H4: MỘT PHẦN. cons4 18 (vs 1–2 ở 0.5B), distribution cân hơn, nhưng r2 vẫn
  dip (20/37) và B-rate ~0.4 — bias giảm chứ chưa hết.

## 6. Caveats (cập nhật: GSM đã fix)
- GSM ban đầu 3–4/50 là EVALUATOR BUG (extractor chỉ chờ `####`, model ra `\boxed{}`).
  Sau fix (boxed/####/last-number + instruction + 512 tokens): s0 35, s20 32, s30 29.
  Retention GSM: s20 91%, s30 83% — khớp câu chuyện free/perm.
- Single run; 9A đã chứng minh deterministic trên 0.5B-V2.
- Calibration WikiText-32 (deviation khỏi paper C4-128) áp dụng đều hai scale nên
  so sánh tương đối vẫn hợp lệ.

## 7. Kết luận và quyết định 3B
Pruning tolerance TĂNG theo scale: cliff dịch từ ~30% (0.5B) lên cao hơn ở 1.5B
(s30 vẫn giữ 76% free). Trend 0.5→1.5 đơn điệu tốt → 3B rất đáng chạy để kiểm tra
monotonic, kỳ vọng s30 giữ ≥85% free ở 3B.
Tiếp theo: tải 3B FP16 (~6GB) rồi lặp ma trận s0/s20/s30.

## 8. Files
bench/eval_10d.py | notes/RUN_10D_15B.json | RESULT_10D.md | REPORT này |
models/Qwen2.5-1.5B-FP16 + pruned-s20/s30.
