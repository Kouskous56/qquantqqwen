# Task 8 CLOSE — demo verify bổ sung
- Cold-start: factual-en 3B 5.2s → steady 0.16s; 0.5B 2.3s → 1.02s.
  Quy tắc: warm-up 1 request trước mọi bench/demo.
- Arithmetic 3B với num_predict=400: ra đầy đủ 888, done_reason=stop.
  Demo bị cắt là do num_predict=200. Kết luận: 3B arithmetic ĐÚNG.
- Sweet spots: compression=Q4_K_M, scale-trên-RTX4050=3B. Winner: Qwen2.5-3B Q4.
