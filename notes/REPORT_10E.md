# Báo cáo đầy đủ 10E-scale: Pruning robustness 0.5B → 1.5B → 3B

## 1. Câu hỏi (preregister)
Robustness có tăng đơn điệu theo scale? Hypothesis: R(3B) > R(1.5B) > R(0.5B) ở s30.
KHÔNG đặt threshold cứng — kết quả nào cũng có giá trị.

## 2. Kết quả 3B (frozen harness, greedy)
| 3B | PPL | free/50 | perm/200 | GSM/50 | B-rate | cons4 |
|---|---|---|---|---|---|---|
| s0 | 8.76 | 24 | 152 | 40 | 0.31 | 29 |
| s20 | 8.91 | 27 | 155 | 41 | 0.30 | 31 |
| s30 | 9.31 | 30 | 151 | 37 | 0.26 | 29 |

## 3. Ma trận retention 3 scale
| Scale | s20 free | s30 free | s30 perm | s30 GSM | ΔPPL s30 |
|---|---|---|---|---|---|
| 0.5B | 100% | 50% | ~88% | - | +7.2% |
| 1.5B | 100% | 76% | 91% | 83% | +6.0% |
| 3B | 112% | 125% | 99% | 92.5% | +6.3% |

## 4. Diễn giải
- Monotonic CONFIRMED, thậm chí mạnh hơn kỳ vọng: 50% → 76% → 125%.
- Free s30 3B (30) vượt s0 (24): nằm trong noise 50 samples HOẶC pruning-regularization.
  Không claim "pruning cải thiện model" — chỉ claim "không degradation đo được".
- ΔPPL s30 ~6-7% cả 3 scale, behavior 50% → 76% → 125%: mapping PPL→behavior
  phụ thuộc scale (khẳng định H3 lần 3).
- B-rate giảm dần theo scale (0.89 → 0.4 → 0.26): bias vị trí cũng scale-dependent.
- Cliff: 0.5B ~30%, 1.5B >30%, 3B chưa thấy cliff ở s30 — cần s40/s50 sau này.

## 5. Kết luận
Pruning robustness TĂNG ĐƠN ĐIỆU theo scale dưới controlled Wanda setup.
Small models are disproportionately fragile — câu chuyện trung tâm đã đủ 3 điểm dữ liệu.

## 6. Files
bench/eval_10e.py | notes/RUN_10E_3B.json | EVAL10E_S*.txt | REPORT này |
models/Qwen2.5-3B-FP16 + pruned-s20/s30.
