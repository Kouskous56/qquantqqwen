"""V2-template cho 2 ban pruned s30 (F16 + Q4 hybrid)."""
from llama_cpp import Llama
import json, re
from collections import Counter
P = "D:/qwen/models/Qwen2.5-0.5B-pruned-s30/"
MODELS = {"s30-F16": P + "s30-f16.gguf", "s30-Q4": P + "s30-q4_k_m.gguf"}
src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(cid, q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for cid, _, q, opts, ans in pat.findall(src)]

def ask(llm, body, mx):
    return llm.create_chat_completion(
        messages=[{"role": "user", "content": body}],
        max_tokens=mx, temperature=0.0)["choices"][0]["message"]["content"]

def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())

llms = {n: Llama(model_path=p, n_ctx=1024, verbose=False) for n, p in MODELS.items()}
R = {n: {"perm": 0, "letters": [], "free": 0, "cons": 0} for n in MODELS}
per_q = {n: {} for n in MODELS}
for ci, (cid, q, opts, cidx) in enumerate(Qs):
    for r in range(4):
        order = [opts[(i + r) % 4] for i in range(4)]
        exp = "ABCD"[(cidx - r) % 4]
        p = (f"Question: {q}\nA) {order[0]}\nB) {order[1]}\nC) {order[2]}\nD) {order[3]}\n"
             "Answer with only the letter A, B, C or D:")
        for n, llm in llms.items():
            L = letter(ask(llm, p, 5))
            R[n]["perm"] += (L == exp)
            R[n]["letters"].append(L)
            per_q[n].setdefault(cid, {})[r] = (L == exp)
    fp = f"{q} Answer with only the value, no explanation:"
    expv = norm(opts[cidx])
    for n, llm in llms.items():
        t = ask(llm, fp, 30)
        R[n]["free"] += (bool(expv) and expv in norm(t))
    print(f"done {ci+1}/50", flush=True)
for n in MODELS:
    c = Counter(R[n]["letters"])
    R[n]["cons"] = sum(1 for cid in per_q[n] if all(per_q[n][cid][r] for r in range(4)))
    print(f"{n}: perm {R[n]['perm']}/200 B-rate {c['B']/200:.2f} "
          f"cons4 {R[n]['cons']}/50 free {R[n]['free']}/50 dist {dict(c)}")
json.dump({n: {"perm": R[n]["perm"], "free": R[n]["free"], "cons": R[n]["cons"],
               "dist": dict(Counter(R[n]["letters"]))} for n in MODELS},
          open("D:/qwen/notes/TEST_HYBRID_S30.json", "w"), indent=1)
print("saved")
