"""Phet tach harness: cung mot model, cung mot runtime, do lai do chinh xac.

Cau hoi: 10pp con lai giua FP16 (transformers) va Q4 (Ollama) la do
  (a) luong tu that su, hay
  (b) khac biet giua hai runtime?

Cach tra loi: chay chinh model dense o do chinh xac F16 (khong mat
thong tin) qua Ollama -- cung harness voi phan Q4. Neu F16-Ollama ra
72% thi do la luong tu that su; neu ra ~62% thi do la harness.
"""
import argparse
import json
import re
import time
import urllib.request

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="qwen3-f16-probe")
_ap.add_argument("--n", type=int, default=50)
_ap.add_argument("--offset", type=int, default=0)
_ap.add_argument("--tag", default="F16PROBE")
_args = _ap.parse_args()

MODEL = _args.model
N = _args.n
OFF = _args.offset
CKPT = f"D:/qwen/notes/R2CKPT_{_args.tag}_{OFF}.json"
OUT = f"D:/qwen/notes/RUN_HARNESS_{_args.tag}_{OFF}.json"


def chat(content, n):
    body = json.dumps({"model": MODEL, "stream": False,
                       "messages": [{"role": "user", "content": content}],
                       "options": {"num_predict": n, "temperature": 0}}
                      ).encode()
    q = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(q, timeout=900) as r:
        return json.load(r)["message"]["content"]


def extract(t):
    mm = re.search(r"\\boxed\{([^}]+)\}", t)
    if mm:
        return mm.group(1).replace(",", "").strip()
    mm = re.search(r"####\s*(-?[\d,.]+)", t)
    if mm:
        return mm.group(1).replace(",", "").strip()
    mg = re.findall(r"-?\d[\d,.]*", t)
    return mg[-1].replace(",", "") if mg else ""


import datasets
gsm = datasets.load_dataset("openai/gsm8k", "main", split="test")
try:
    det = json.load(open(CKPT, encoding="utf-8"))["details"]
    print(f"resume: {len(det)}/{N}", flush=True)
except (FileNotFoundError, KeyError, ValueError):
    det = []
t0 = time.time()
for j in range(N):
    if j < len(det):
        continue
    i = OFF + j
    r = gsm[i]
    t = chat(r["question"] + "\nThink step by step, then end with: "
             "#### <number>", 256)
    me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
    exp = me.group(1).replace(",", "") if me else ""
    det.append({"i": i, "pass": extract(t) == exp})
    json.dump({"done": len(det), "details": det}, open(CKPT, "w"))
    if len(det) % 20 == 0:
        print(f"  {len(det)}/{N} ({time.time()-t0:.0f}s)", flush=True)

s = sum(d["pass"] for d in det)
print(f"F16-Ollama [{OFF}..{OFF+N-1}]: {s}/{len(det)} = {100*s/len(det):.1f}%")
json.dump({"model": MODEL, "n": len(det), "offset": OFF, "score": s,
           "acc": round(s / max(1, len(det)), 4), "details": det,
           "precision": "F16 lossless, cung runtime Ollama",
           "elapsed_s": round(time.time() - t0)}, open(OUT, "w"), indent=1)
print(f"saved {OUT}")
