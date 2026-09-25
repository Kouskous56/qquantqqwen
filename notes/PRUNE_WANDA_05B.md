# Task 7 — Wanda pruning Qwen2.5-0.5B (PPL wikitext 200 dòng, baseline 13.27)
- s20 unstructured: 13.51 (+0.24) — TỐT
- s30 unstructured: 14.23 (+0.96) — CHẤP NHẬN ĐƯỢC
- s50 unstructured: 22.13 (+8.86) — SẬP
- s50 2:4 structured: 63.92 — HỦY (2:4 cứng cần weight-update kiểu SparseGPT)
Kết luận: 0.5B chịu tối đa ~30% unstructured. 2:4 để dành 1.5B+ hoặc kèm SparseGPT.
Code: scripts/wanda_qwen.py (Wanda gốc, calibration wikitext-32 thay vì C4-128 của paper).
