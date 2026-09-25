# Task 6 — chọn preset cho 3B (prompt photosynthesis, GPU)
- F16 5.9GB | Q8 3.1GB | Q5_K_M 2.1GB | Q4_K_M 1.8GB | Q3_K_M 1.5GB
- mine-Q4 (GPU 2.2GB): chi tiết, chuẩn ~ bản official
- mine-Q3 (GPU 1.8GB): đúng nhưng sơ lược, mất chi tiết
- official qwen2.5:3b: chi tiết như Q4
Kết luận: Q4_K_M thắng — giảm 70% size (5.9GB→1.8GB), giữ chất lượng. Q5 nếu cần thêm chút, Q3 loại.
