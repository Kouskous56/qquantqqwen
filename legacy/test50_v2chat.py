"""V2 rerun cho 0.5B VOI chat template (fix pipeline bug)."""
from llama_cpp import Llama
import json, time, re
BASE = "D:/qwen/models/Qwen2.5-0.5B/"
MODELS = {"F16": BASE + "my-qwen05-f16.gguf", "Q8": BASE + "my-qwen05-q8_0.gguf",
          "Q4": BASE + "my-qwen05-q4_k_m.gguf", "Q3": BASE + "my-qwen05-q3_k_m.gguf"}
src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(cid, band, q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for cid, band, q, opts, ans in pat.findall(src)]

def ask(llm, body, mx):
    t = llm.create_chat_completion(
        messages=[{"role": "user", "content": body}],
        max_tokens=mx, temperature=0.0)["choices"][0]["message"]["content"]
    return t

def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())

llms = {n: Llama(model_path=p, n_ctx=1024, verbose=False) for n, p in MODELS.items()}
R = {n: {"perm_correct": 0, "letters": [], "free_correct": 0, "raw": {}} for n in MODELS}
for ci, (cid, band, q, opts, cidx) in enumerate(Qs):
    for r in range(4):
        order = [opts[(i + r) % 4] for i in range(4)]
        exp = "ABCD"[(cidx - r) % 4]
        p = (f"Question: {q}\nA) {order[0]}\nB) {order[1]}\nC) {order[2]}\nD) {order[3]}\n"
             "Answer with only the letter A, B, C or D:")
        for n, llm in llms.items():
            t = ask(llm, p, 5)
            L = letter(t)
            R[n]["perm_correct"] += (L == exp)
            R[n]["letters"].append(L)
            R[n]["raw"].setdefault(cid, {})[f"r{r}"] = {"got": L, "exp": exp, "out": t[:120]}
    fp = f"{q} Answer with only the value, no explanation:"
    expv = norm(opts[cidx])
    for n, llm in llms.items():
        t = ask(llm, fp, 30)
        R[n]["free_correct"] += (bool(expv) and expv in norm(t))
        R[n]["raw"][cid]["free"] = t[:150]
    print(f"done {ci+1}/50", flush=True)
from collections import Counter
for n in MODELS:
    c = Counter(R[n]["letters"])
    cons = sum(1 for cid in R[n]["raw"] if all(
        R[n]["raw"][cid][f"r{r}"]["got"] == R[n]["raw"][cid][f"r{r}"]["exp"] for r in range(4)))
    print(f"{n}: perm {R[n]['perm_correct']}/200 B-rate {c['B']/200:.2f} "
          f"cons4 {cons}/50 free {R[n]['free_correct']}/50 dist {dict(c)}")
json.dump(R, open("D:/qwen/notes/TEST50_V2_CHAT.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved")
