"""Task 9B: dense -> s20 -> s20+Q4, frozen harness (giong 9A) + perm 1x."""
from llama_cpp import Llama
import json, time, re
from collections import Counter
P20 = "D:/qwen/models/Qwen2.5-0.5B-pruned-s20/"
BASE = "D:/qwen/models/Qwen2.5-0.5B/"
MODELS = {"dense-F16": BASE + "my-qwen05-f16.gguf",
          "dense-Q4": BASE + "my-qwen05-q4_k_m.gguf",
          "s20-F16": P20 + "s20-f16.gguf", "s20-Q4": P20 + "s20-q4_k_m.gguf"}
PARAMS = {"n_ctx": 1024, "max_tokens": 30, "temperature": 0.0}
src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(cid, q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for cid, _, q, opts, ans in pat.findall(src)]

def ask(llm, body, mx=30):
    return llm.create_chat_completion(
        messages=[{"role": "user", "content": body}],
        max_tokens=mx, temperature=PARAMS["temperature"])["choices"][0]["message"]["content"]

def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())

llms = {n: Llama(model_path=p, n_ctx=PARAMS["n_ctx"], verbose=False) for n, p in MODELS.items()}
R = {}
for n, llm in llms.items():
    free_reps, letters = [], []
    for rep in range(3):
        ok = 0
        for cid, q, opts, cidx in Qs:
            t = ask(llm, f"{q} Answer with only the value, no explanation:")
            expv = norm(opts[cidx])
            ok += (bool(expv) and expv in norm(t))
        free_reps.append(ok)
    for cid, q, opts, cidx in Qs:
        for r in range(4):
            order = [opts[(i + r) % 4] for i in range(4)]
            exp = "ABCD"[(cidx - r) % 4]
            p = (f"Question: {q}\nA) {order[0]}\nB) {order[1]}\nC) {order[2]}\nD) {order[3]}\n"
                 "Answer with only the letter A, B, C or D:")
            letters.append(letter(ask(llm, p, 5)))
    c = Counter(letters)
    perm = sum(1 for cid_i, (cid, q, opts, cidx) in enumerate(Qs) for r in range(4)
               if letters[cid_i * 4 + r] == "ABCD"[(cidx - r) % 4])
    R[n] = {"free_reps": free_reps, "perm": perm, "b_rate": round(c["B"] / 200, 3)}
    print(f"{n}: free {free_reps} perm {perm}/200 B-rate {R[n]['b_rate']}", flush=True)
json.dump({"run_id": "9B-" + time.strftime("%Y%m%d-%H%M"), "params": PARAMS, "results": R},
          open("D:/qwen/notes/RUN_9B.json", "w"), indent=1)
print("saved RUN_9B.json")
