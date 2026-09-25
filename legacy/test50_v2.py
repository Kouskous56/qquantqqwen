"""Benchmark v2 chong bias: permutation 4 chieu + free-response, log raw.
Models: F16, Q4 (0.5B CPU) + mine-q4, official (3B Ollama GPU)."""
from llama_cpp import Llama
import json, time, urllib.request, re

BASE = "D:/qwen/models/Qwen2.5-0.5B/"
LOCAL = {"F16": BASE + "my-qwen05-f16.gguf", "Q4": BASE + "my-qwen05-q4_k_m.gguf"}
API = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODELS = ["qwen3b-mine-q4", "qwen2.5:3b"]

src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(cid, band, q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for cid, band, q, opts, ans in pat.findall(src)]
assert len(Qs) == 50

def ask_local(llm, prompt, mx=5):
    t = llm(prompt, max_tokens=mx, temperature=0.0)["choices"][0]["text"]
    return t

def ask_ollama(model, prompt, mx=5):
    req = urllib.request.Request(API, data=json.dumps(
        {"model": model, "prompt": prompt, "stream": False,
         "options": {"num_ctx": 4096, "num_predict": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["response"]

def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

def mcq_prompt(q, opts):
    return (f"Question: {q}\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\n"
            "Answer with only the letter A, B, C or D:")

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())

llms = {n: Llama(model_path=p, n_ctx=512, verbose=False) for n, p in LOCAL.items()}
R = {}
for name in list(LOCAL) + OLLAMA_MODELS:
    R[name] = {"perm_correct": 0, "perm_total": 0, "consistent": 0,
               "letters": [], "free_correct": 0, "raw": {}}

for ci, (cid, band, q, opts, cidx) in enumerate(Qs):
    # 4 rotations: correct value lan luot o A,B,C,D
    for r in range(4):
        order = [opts[(i + r) % 4] for i in range(4)]
        exp_letter = "ABCD"[(cidx - r) % 4]
        exp_val = opts[cidx]
        p = mcq_prompt(q, order)
        for n, llm in llms.items():
            t = ask_local(llm, p)
            L = letter(t)
            R[n]["perm_total"] += 1
            R[n]["perm_correct"] += (L == exp_letter)
            R[n]["letters"].append(L)
            R[n]["raw"].setdefault(cid, {})[f"r{r}"] = {"got": L, "exp": exp_letter, "out": t[:120]}
        for m in OLLAMA_MODELS:
            t = ask_ollama(m, p)
            L = letter(t)
            R[m]["perm_total"] += 1
            R[m]["perm_correct"] += (L == exp_letter)
            R[m]["letters"].append(L)
            R[m]["raw"].setdefault(cid, {})[f"r{r}"] = {"got": L, "exp": exp_letter, "out": t[:120]}
    # free-response
    fp = f"{q} Answer with only the value, no explanation:"
    expv = norm(opts[cidx])
    for n, llm in llms.items():
        t = ask_local(llm, fp, mx=30)
        if expv and expv in norm(t):
            R[n]["free_correct"] += 1
        R[n]["raw"][cid]["free"] = t[:150]
    for m in OLLAMA_MODELS:
        t = ask_ollama(m, fp, mx=30)
        if expv and expv in norm(t):
            R[m]["free_correct"] += 1
        R[m]["raw"][cid]["free"] = t[:150]
    print(f"done {ci+1}/50 {cid}", flush=True)

# consistency: cung value dung o ca 4 rotations
for n in R:
    from collections import Counter
    R[n]["b_rate"] = round(Counter(R[n]["letters"])["B"] / len(R[n]["letters"]), 3)
    cons = 0
    for cid in R[n]["raw"]:
        oks = [R[n]["raw"][cid].get(f"r{r}", {}).get("got") == R[n]["raw"][cid].get(f"r{r}", {}).get("exp")
               for r in range(4)]
        if all(oks):
            cons += 1
    R[n]["consistent_4of4"] = cons

out = {n: {k: v for k, v in R[n].items() if k != "raw"} for n in R}
print(json.dumps(out, indent=1))
json.dump(R, open("D:/qwen/notes/TEST50_V2.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
L = ["# Benchmark v2: permutation + free-response", ""]
for n, d in out.items():
    L.append(f"## {n}: perm_acc {d['perm_correct']}/{d['perm_total']} "
             f"consistent4/4 {d['consistent_4of4']}/50 B-rate {d['b_rate']} free {d['free_correct']}/50")
open("D:/qwen/notes/TEST50_V2.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("saved")
