# So sánh chất lượng quant 0.5B — prompt "The capital of France is", temp 0
- F16 948MB: "...Paris. It is the largest city in Europe..." — OK
- Q8_0 506MB: OK, gần như F16
- Q4_K_M 379MB: OK, giữ chất lượng tốt, giảm 60% size
- Q3_K_M 339MB: LẶP CÂU "Paris is the capital of France." ×3 — hỏng ở 3-bit
Kết luận: Q4_K_M là preset tối ưu cho edge. Từ Q3 trở xuống rủi ro lặp câu.
