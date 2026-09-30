"""GSM-200 cho cac ban Q5_K_M / Q6_K (cung 200 cau dau, cung harness)."""
import argparse
import json
import re
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--precision", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--offset", type=int, default=0)
a = ap.parse_args()

CK = f"D:/qwen/notes/R2CKPT_GSMSW_{a.tag}_{a.offset}.json"
OUT = f"D:/qwen/notes/RUN_GSMSWEEP_{a.tag}_{a.offset}.json"


def chat(content, n):
    body = json.dumps({"model": a.model, "stream": False,
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
    det = json.load(open(CK, encoding="utf-8"))["details"]
except (FileNotFoundError, KeyError, ValueError):
    det = []
t0 = time.time()
for j in range(a.n):
    if j < len(det):
        continue
    i = a.offset + j
    r = gsm[i]
    t = chat(r["question"] + "\nThink step by step, then end with: "
             "#### <number>", 256)
    me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
    exp = me.group(1).replace(",", "") if me else ""
    # Luu ca exp va got: hash noi dung dataset chi pin duoc chi so i neu
    # khong co chung. Ban truoc chi ghi {"i","pass"} nen manifest chi
    # chung minh duoc "dung 400 cau nao", khong chung minh duoc cau do co
    # gi. Xem notes/QWEN3_BLOCK.md.
    det.append({"i": i, "exp": exp, "got": extract(t), "pass": extract(t) == exp})
    json.dump({"done": len(det), "details": det}, open(CK, "w"))
    if len(det) % 25 == 0:
        print(f"  {len(det)}/{a.n} ({time.time()-t0:.0f}s)", flush=True)

s = sum(d["pass"] for d in det)
json.dump({"model": a.model, "tag": a.tag, "precision": a.precision,
           "n": len(det), "gsm": s, "acc": round(s / max(1, len(det)), 4),
           "offset": a.offset, "details": det,
           "elapsed_s": round(time.time() - t0)}, open(OUT, "w"), indent=1)
print(f"GSM-{a.n} [{a.precision}] {a.tag}: {s}/{len(det)} = "
      f"{100*s/len(det):.1f}%  ({time.time()-t0:.0f}s)")
print(f"saved {OUT}")
