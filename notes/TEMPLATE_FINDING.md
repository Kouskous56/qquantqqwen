# Template finding — collapse có thật, hướng collapse phụ thuộc format
## Thí nghiệm (10 câu × 4 rotations, 0.5B-F16):
- raw prompt: acc 10/40, B-rate 1.00, single-char 0/40
- chat template: acc 16/40, B-rate 0.28, single-char 40/40
## V2 rerun full VỚI template (TEST50_V2_CHAT.json):
- F16: perm 64/200 (32%), phân bố A74/B58/C63/D5, cons4 2/50, free 18/50
- Q8: 70/200 (35%), cons4 2/50, free 18/50
- Q4: 61/200, A-collapse (A155!), cons4 1/50, free 15/50
- Q3: 57/200, A-collapse (A179!), cons4 1/50, free 11/50
## Kết luận:
- Format compliance FIX được bằng template (single-char trở lại)
- Nhưng positional collapse là THẬT (đổi hướng theo format: B khi raw → A khi templated)
- Capability 0.5B ~30% cả 2 format, quant không làm tệ thêm đáng kể
- 3B (đã dùng template qua Ollama từ đầu) miễn nhiễm vấn đề này
