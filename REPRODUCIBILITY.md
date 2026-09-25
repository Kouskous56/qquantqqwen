# REPRODUCIBILITY — claim → command → manifest → expected output
Models (download once from Hugging Face): Qwen2.5-{0.5B,1.5B,3B}-Instruct.
Set MODELROOT to your weights directory. All commands run on CUDA.

## Standardized PPL (Claim: ΔPPL@s30 ≈ +6–8% all scales)
```bash
python reproduce/eval_ppl.py --model $MODELROOT/Qwen2.5-0.5B-FP16 \
  --tok $MODELROOT/Qwen2.5-0.5B-FP16 --corpus bench-corpus/wikitext_test.txt \
  --out results/ppl_05B_s0.json
```
Expected: `05B-s0 ≈ 15.11`, `05B-s30 ≈ 16.31`, `15B-s0 ≈ 9.93`,
`15B-s30 ≈ 10.58`, `3B-s0 ≈ 8.76`, `3B-s30 ≈ 9.31` (see `results/standardized_ppl.csv`).

## V2 behavioral eval — quantization agreement (legacy GGUF block)
```bash
python reproduce/eval_v2.py --model model-q4.gguf \
  --questions data/v2/questions.json --use-template --out results/v2_3b.json
```
Expected (3B self-Q4): `perm ≈ 149/200`, `free ≈ 37/50`.

## V2 behavioral eval — cross-scale pruning (HF transformers block)
```bash
python reproduce/eval_v2.py --model <pruned-or-dense-HF-dir> ...
```
Same script, same 50 questions; do not mix GGUF-block and HF-block numbers.
Expected retention: 0.5B 9/17 (53%), 1.5B 29/38 (76%), 3B 30/24.
Manifests: `manifests/RUN_05B_S30_FREE.json`, `RUN_9B.json`, `RUN_10D_15B.json`.

## GSM8K held-out (Claim: recovery is domain-specific)
```bash
python reproduce/eval_gsm.py --model $MODELROOT/<pruned-or-recovered> --n 50 --out results/gsm.json
```
Expected: s30-0.5B base 4/50 (sparse), GSM-masked-recovery 14/50.

## Wanda pruning (Claim: s20 safe, s30 boundary at 0.5B)
```bash
python reproduce/prune_wanda.py --src $MODELROOT/Qwen2.5-0.5B-FP16 \
  --out $MODELROOT/pruned-s30 --sparsity 0.3
```

## Masked recovery (Claim: sparsity preserved + domain recovery)
```bash
python reproduce/recover_masked.py --src $MODELROOT/pruned-s30 --data gsm8k \
  --out-merged $MODELROOT/rec-merged --out-masked $MODELROOT/rec-masked
```
Expected: merged sparsity → 0.00, masked stays 0.30.

## Manifests
Every decisive run stores versions, dataset hashes and run IDs — see `manifests/`.
Legacy development scripts live in `legacy/`; canonical entry points are `reproduce/`.
