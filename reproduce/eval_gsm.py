"""Canonical GSM8K held-out evaluator (fixed extractor: boxed -> #### -> number).
Usage:
  python reproduce/eval_gsm.py --model <hf-dir> --n 50 --out out.json
"""
import argparse
import json
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract(t):
    m = re.search(r"\\boxed\{([^}]+)\}", t)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"####\s*(-?[\d,.]+)", t)
    if m:
        return m.group(1).replace(",", "")
    mg = re.findall(r"-?\d[\d,.]*", t)
    return mg[-1].replace(",", "") if mg else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tok or a.model)
    m = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, device_map="cuda",
        low_cpu_mem_usage=True).eval()
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
    ok, raws = 0, []
    with torch.no_grad():
        for r in ds:
            body = tok.apply_chat_template(
                [{"role": "user",
                  "content": r["question"] + "\nThink step by step, then end with: #### <number>"}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(body, return_tensors="pt").to("cuda")
            out = m.generate(**ids, max_new_tokens=a.max_tokens, do_sample=False,
                             eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id)
            t = tok.decode(out[0][ids.input_ids.shape[1]:])
            me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
            exp = me.group(1).replace(",", "") if me else ""
            got = extract(t)
            ok += (got == exp)
            raws.append({"exp": exp, "got": got})
    res = {"model": a.model, "n": a.n, "correct": ok,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M")}
    json.dump({"summary": res, "items": raws}, open(a.out, "w"), indent=1)
    print(f"gsm {ok}/{a.n}")


main()
