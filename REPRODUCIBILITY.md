# REPRODUCIBILITY.md — claim → script → manifest → output

## Claim 1: standardized PPL across scales
- Script: `bench/ppl_std.py` (512/256, token-weighted, no template, FP16)
- Models (not included, see README): Qwen2.5-{0.5B,1.5B,3B}-Instruct + pruned variants
- Manifest: `manifests/RUN_PPL_STD.json`
- Results: `results/standardized_ppl.csv`, `notes/PPL_STD_TABLE.md`

## Claim 2: s30 behavioral retention across scales
- Script: `bench/test_9b.py`-style frozen V2 (see `bench/test50_v2chat.py`, `bench/eval_10d.py`, `bench/eval_10e.py`)
- Manifests: `manifests/RUN_9B.json`, `RUN_10D_15B.json`, `RUN_10E_3B.json`
- Results: `results/pruning_retention_05B.csv`, `results/scale_retention.csv`

## Claim 3: masked + domain-matched recovery
- Scripts: `scripts/recover_lora.py`, `scripts/recover_gsm8k.py`,
  `bench/eval_9c.py`, `bench/eval_10c.py`
- Manifests: `notes/RECOVERY_9C.md`, `notes/REPORT_10C.md`
- Results: `notes/RESULT_10C.md` (C0=4/C1=3/C2=14 on GSM-heldout/50)

## Claim 4: Q4 behavioral preservation
- Notes: `notes/QUANT_COMPARE_05B.md`, `QUANT_PRESET_3B.md`, `TEST50_V2*.json` (in `notes/` of dev machine; aggregates in paper)
- 2:4 negative result: `notes/RESULT_11B1.md`, `third_party_adapters/sparsegpt_qwen.py`

## Claim 5: deployment + realized benefit
- Bench: `bench/bench_gpu.py`, `bench/demo_chatbot.py`, `bench/measure_11a.py`
- Notes: `notes/BENCH_BASELINE.md`, `DEMO_CHATBOT.md`, `RESULT_11A.md`

## Environment
Python 3.11, torch 2.9.0+cu129, transformers 5.17.0, RTX 4050 6GB, native Windows.
See `requirements.txt`. Greedy decoding throughout; frozen harnesses log manifests.
