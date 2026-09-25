# Hybrid s30+Q4 vs dense (V2-template, 0.5B)
| Biến thể | perm/200 | free/50 | cons4 | PPL trước đó |
|---|---|---|---|---|
| dense F16 | 64 | 18 | 2 | 13.27 |
| dense Q4 | 61 | 15 | 1 | - |
| s30 F16 | 56 | 6 | 1 | 14.23 |
| s30+Q4 | 58 | 8 | 1 | - |
Kết luận: prune 30% giữ MCQ (~-8 điểm cùng precision) nhưng HỦY free-generation (18→6).
PPL +0.96 không phản ánh đủ. Chatbot edge cần generation → s30 ĐẮT.
Để chatbot dùng được: giữ dense-Q4, hoặc prune nhẹ 20% + LoRA hồi phục.
