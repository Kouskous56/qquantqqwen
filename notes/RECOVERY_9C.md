# 9C — masked recovery (s20 + LoRA rank 8, 300 samples wikitext, 1 epoch)
- LoRA train loss 3.23 → 2.98
- Merge thường: sparsity 0.20 → 0.00 (MẤT sparsity), PPL 12.20
- Masked merge: sparsity GIỮ 0.20, PPL 12.13, free-response 17/50 (~18 base)
- Caveat: PPL cải thiện một phần do overfit domain (train=test=wikitext)
Trả lời: CÓ, masked recovery giữ sparsity + hồi phục generation.
Adapter: models/s20-lora-adapter | masks: models/s20-masks.pt
Lưu ý khoa học: s20 chưa hỏng trước recovery (18→17) nên đây mới là
"preserve", chưa phải "recover" — xem 10B/10C cho recovery thật.
