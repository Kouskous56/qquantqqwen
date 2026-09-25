"""10A: V2-template cho artifact cuoi s20-masked-Q4 (frozen harness)."""
from llama_cpp import Llama
import json, re
from collections import Counter
P = "D:/qwen/models/s20-lora-masked/s20r-q4_k_m.gguf"
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

llm = Llama(model_path=P, n_ctx=1024, verbose=False)
letters, per_q, free = [], {}, 0
for ci, (cid, q, opts, cidx) in enumerate(Qs):
    for r in range(4):
        order = [opts[(i + r) % 4] for i in range(4)]
        exp = "ABCD"[(cidx - r) % 4]
        p = (f"Question: {q}\nA) {order[0]}\nB) {order[1]}\nC) {order[2]}\nD) {order[3]}\n"
             "Answer with only the letter A, B, C or D:")
        L = letter(ask(llm, p, 5))
        letters.append(L)
        per_q.setdefault(cid, {})[r] = (L == exp)
    fp = f"{q} Answer with only the value, no explanation:"
    expv = norm(opts[cidx])
    free += (bool(expv) and expv in norm(ask(llm, fp, 30)))
    print(f"done {ci+1}/50", flush=True)
c = Counter(letters)
perm = sum(1 for cid in per_q for r in range(4) if per_q[cid][r])
cons = sum(1 for cid in per_q if all(per_q[cid][r] for r in range(4)))
print(f"s20r-Q4: perm {perm}/200 B-rate {c['B']/200:.3f} cons4 {cons}/50 free {free}/50 dist {dict(c)}")
json.dump({"s20r-Q4": {"perm": perm, "free": free, "cons": cons, "dist": dict(c)}},
          open("D:/qwen/notes/RUN_10A.json", "w"), indent=1)
print("saved RUN_10A.json")
