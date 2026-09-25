# Báo cáo chuyên sâu: External Benchmark Pack (lm-eval harness)

## 1. Động cơ
V2 tự soạn bị nghi "tự ra đề tự chấm". Cần benchmark ngoài độc lập kiểm chứng:
positional bias, instruction compliance, và retention sau compression.

## 2. Tooling (vật lộn đáng ghi vào methods)
- lm-eval 0.4.13 + backend hf; Windows console cp1252 crash bảng kết quả → PYTHONUTF8=1.
- batch auto CUBLAS-fail, batch 8 OOM → chốt batch 2 + expandable_segments.
- IFEval qua HF backend quá chậm (160s/it, ETA 24h) → chuyển Ollama local-completions.
- local-completions 400 liên tiếp, root-cause lần lượt: CLI bool-string bug
  (tokenized_requests='False' truthy → prompt array), stop:[null] (thiếu eos_string),
  batch>1 (Ollama không nhận prompt array).
- NLTK punkt_tab bị chặn → tải thủ công từ GitHub raw.
- Runner Ollama crash ở doc 487/541 → custom resilient loop (skip-on-error,
  incremental save) + chấm bằng harness process_results.
- Zombie processes dọn 2 lần.

## 3. MetaBench 0.5B dense (858 items, loglikelihood)
overall 0.3191 | arc .297 | gsm flex .287/strict .203 | hellaswag .183
(dưới chance — nghi template mismatch) | mmlu .448 | truthfulqa .149 | winogrande .519.
Permute: overall 0.3393; arc/hellaswag Y HỆT bản gốc (đã verify permutation thật).

## 4. Finding phương pháp quan trọng
lm-eval chấm bằng LOGLIKELIHOOD từng choice; V2 chấm bằng GENERATION chữ cái.
Bias vị trí sống ở generative selection, không hiện trong likelihood scoring.
Hai method đo hai thứ khác nhau — paper phải tách bạch, không được claim
"MetaBench bác bỏ V2" hay ngược lại.

## 5. IFEval qwen3b-mine-q4 (541 prompts, Ollama)
prompt-strict 0.6007 | loose 0.6414 | inst-strict 0.6882 | inst-loose 0.7242.
Strict 60% rất mạnh cho 3B-Q4 tự quant; khớp demo qualitative.
Responses thô đã lưu để audit.

## 6. Còn thiếu (chưa chạy)
MetaBench cho s20/s30/pruned các scale, IFEval cho s30/Ollama-official,
WikiText chuẩn hóa, GSM8K-full 1319, MMLU-Pro tier 2.

## 7. Files
bench/run_ifeval_*.py | scripts/debug_*.py | notes/lmeval/* (results+samples) |
MANIFEST_MB05D/P.md, MANIFEST_IFEVAL_MINEQ4.md | BENCH_MANIFEST_TEMPLATE.md.
