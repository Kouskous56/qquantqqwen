"""Qwen3-4B FP16 GSM-50 only, max 256 tokens (device_map auto, offload OK)."""
import argparse
import json
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model-root", default="D:/qwen/models/Qwen3-4B-FP16",
                help="thu muc checkpoint (mac dinh giu duong lab cu)")
ap.add_argument("--notes-dir", default="D:/qwen/notes",
                help="thu muc ghi manifest (mac dinh giu duong lab cu)")
ap.add_argument("--n", type=int, default=50)
a = ap.parse_args()

P = a.model_root
NOTES = a.notes_dir.rstrip("/\\")
tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16,
                                         device_map="auto").eval()


@torch.no_grad()
def gen(body, mx=256):
    ids = tok(body, return_tensors="pt").to("cuda")
    out = m.generate(**ids, max_new_tokens=mx, do_sample=False,
                     eos_token_id=tok.eos_token_id,
                     pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:])


def extract(t):
    mm = re.search(r"\\boxed\{([^}]+)\}", t)
    if mm:
        return mm.group(1).replace(",", "").strip()
    mm = re.search(r"####\s*(-?[\d,.]+)", t)
    if mm:
        return mm.group(1).replace(",", "")
    mg = re.findall(r"-?\d[\d,.]*", t)
    return mg[-1].replace(",", "") if mg else ""


def chat(q):
    return tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)


ds = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
g, det = 0, []
t0 = time.time()
for i, r in enumerate(ds):
    t = gen(chat(r["question"] + "\nThink step by step, then end with: #### <number>"))
    me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
    exp = me.group(1).replace(",", "") if me else ""
    ok = extract(t) == exp
    g += ok
    det.append({"i": i, "pass": bool(ok)})
    print(f"gsm {i}: {ok} ({time.time()-t0:.0f}s)", flush=True)
json.dump({"model": "Qwen3-4B-Instruct-2507-FP16", "gsm50_256": g,
           "details": det, "elapsed_s": round(time.time() - t0),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M")},
          open(f"{NOTES}/RUN_QWEN3_GSM.json", "w"), indent=1)
print(f"GSM-50: {g}/50")
