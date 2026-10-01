"""GSM8K-200 qua Ollama (GGUF artifact), checkpoint + resume.

Protocol: greedy (temp 0), num_predict 256, prompt CoT + "#### <number>".
Lay tu dong: dung 200 cau dau cua split test (co dinh --seed offset).
Score: so sanh so cuoi (extract) voi gold, bo dau phay.
"""
import argparse
import json
import re
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--npredict", type=int, default=256)
args = ap.parse_args()

OUT = f"D:/qwen/notes/RUN_QWEN3_{args.tag}.json"
CKPT = f"D:/qwen/notes/CKPT_GSM200_{args.tag}.json"


def chat(content, n):
    body = json.dumps({"model": args.model, "stream": False,
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
ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
items = [(i, ds[i]) for i in range(args.offset, args.offset + args.n)]

det = []
if __import__("os").path.exists(CKPT):
    det = json.load(open(CKPT))["details"]
    print(f"resume: {len(det)}/{args.n} done", flush=True)

t0 = time.time()
for k, (i, r) in enumerate(items):
    if k < len(det):
        continue
    t = chat(r["question"]
             + "\nThink step by step, then end with: #### <number>", args.npredict)
    me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
    exp = me.group(1).replace(",", "") if me else ""
    got = extract(t)
    ok = got == exp
    det.append({"i": i, "pass": bool(ok), "got": got, "exp": exp,
                "chars": len(t)})
    json.dump({"done": len(det), "n": args.n, "details": det},
              open(CKPT, "w"))
    print(f"gsm {k+1}/{args.n} idx={i}: {ok} got={got} exp={exp} "
          f"({time.time()-t0:.0f}s)", flush=True)

g = sum(d["pass"] for d in det)
json.dump({"model": args.model, "tag": args.tag, "n": len(det),
           "gsm": g, "gsm200": g, "offset": args.offset,
           "protocol": "ollama-greedy-CoT-####-num_predict256",
           "details": det, "elapsed_s": round(time.time() - t0),
           "note": "GGUF/Q4 artifact via Ollama; do not mix voi FP16 GSM-50"},
          open(OUT, "w"), indent=1)
print(f"GSM-{args.n}: {g}/{len(det)} saved {OUT}")
