# 11A — realized benefit: dense vs s30/s40/s50 (transformers FP16, 3 prompts × 100 tok)
| Model | peak VRAM | TTFT | prefill t/s | decode t/s | wall | power | J/tok |
|---|---|---|---|---|---|---|---|
| dense | 5.8GB | .31/.11/.11 | 41.0 | 10.1 | 23.6s | 37.2W | 3.69 |
| s30 | 5.8GB | .14/.11/.24 | 118.6* | 6.1 | 40.0s | 31.2W | 5.15 |
| s40 | 5.8GB | .26/.22/.25 | 56.0 | 4.9 | 48.0s | 29.6W | 6.07 |
| s50 | 5.8GB | .33/.25/.25 | 54.9 | 4.8 | 52.9s | 29.6W | 6.14 |
Trả lời: sparsity KHÔNG tạo benefit nào — VRAM y hệt, decode CHẬM hơn (10.1→4.8),
năng lượng/token TỆ hơn (3.69→6.14 J).
Caveats: *prefill s30 cao do thiếu warm-up control (dense chạy đầu tiên, cold);
output length khác nhau giữa dense (EOS sớm) và pruned (ramble dài) ảnh hưởng wall;
single run. Dù vậy, kết luận định tính vững: zero sparsity → zero memory saving, no speedup.
