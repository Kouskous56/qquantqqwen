"""Canonical PPL evaluator (512/256, token-weighted, no template, FP16, greedy N/A).
Usage:
  python reproduce/eval_ppl.py --model <path> --tok <path> --corpus <txt> --out out.json
"""
import argparse
import hashlib
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    a = ap.parse_args()

    text = open(a.corpus, encoding="utf-8").read()
    tok = AutoTokenizer.from_pretrained(a.tok)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, device_map="cuda",
        low_cpu_mem_usage=True).eval()
    enc = tok(text, return_tensors="pt")
    seq = enc.input_ids.size(1)
    nlls, nt = [], 0
    with torch.no_grad():
        for i in range(0, seq, a.stride):
            b = max(i + a.stride - a.max_len, 0)
            e = min(i + a.stride, seq)
            ids = enc.input_ids[:, b:e].cuda()
            tgt = ids.clone()
            tgt[:, :-a.stride] = -100
            out = m(ids, labels=tgt)
            nlls.append(out.loss * (e - max(b, e - a.stride)))
            nt += e - max(b, e - a.stride)
    ppl = torch.exp(torch.stack(nlls).sum() / nt).item()
    res = {"model": a.model, "corpus_sha": hashlib.sha256(text.encode()).hexdigest()[:12],
           "max_len": a.max_len, "stride": a.stride, "tokens": nt,
           "ppl": round(ppl, 2), "runtime_s": round(time.time() - t0, 1),
           "torch": torch.__version__}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"PPL {ppl:.2f} ({nt} tok)")


main()
