# Benchmark-pack status: MetaBench ngoài + blockers

## Xong
- 0.5B dense MetaBench: overall 0.3191 (arc .297, gsm .287/.203, hellaswag .183,
  mmlu .448, truthfulqa .149, winogrande .519) + permute 0.3393.
- Finding method: likelihood-scoring (lm-eval) vs generation (V2) đo hai thứ khác nhau.
- IFEval mine-Q4: strict 0.60/loose 0.64 (resilient loop, audit 541 unique).
- Tooling fixes: PYTHONUTF8, batch 2, NLTK punkt thủ công, eos/bool/batch bugs của local-completions.

## Blockers 3B-HF MetaBench
- Forward trực tiếp 0.22s nhưng qua harness ~20s/request (batch 2), batch 8 OOM.
- Nghi collator/padding bất thường, chưa xác định — dừng debug, tránh rabbit hole.
- Ollama không trả logprobs → loại backend này cho MetaBench.

## Lối đi
llama-server (đã build) + GGUF cho loglikelihood: retention HYBRID
(mine-Q4 vs s30-Q4), không phải pure-FP16. Cần convert s30-3B → GGUF → Q4.
Pure-3B-FP16 MetaBench ghi limitation.

## Chưa chạy
MetaBench s20/s30 mọi scale, IFEval s30, WikiText chuẩn, GSM-full, MMLU-Pro.
