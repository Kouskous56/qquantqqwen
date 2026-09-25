"""Kiem tra gia thuyet template: 0.5B-F16 voi chat template dung vs raw prompt.
10 cau x 4 rotations = 40 samples."""
from llama_cpp import Llama
import re
llm = Llama(model_path="D:/qwen/models/Qwen2.5-0.5B/my-qwen05-f16.gguf",
            n_ctx=1024, verbose=False)
src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(cid, q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for cid, _, q, opts, ans in pat.findall(src)][:10]

def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

for mode in ["raw", "chat"]:
    ok, bcount, single, tot = 0, 0, 0, 0
    for cid, q, opts, cidx in Qs:
        for r in range(4):
            order = [opts[(i + r) % 4] for i in range(4)]
            exp = "ABCD"[(cidx - r) % 4]
            body = (f"Question: {q}\nA) {order[0]}\nB) {order[1]}\nC) {order[2]}\nD) {order[3]}\n"
                    "Answer with only the letter A, B, C or D:")
            if mode == "raw":
                t = llm(body, max_tokens=5, temperature=0.0)["choices"][0]["text"]
            else:
                t = llm.create_chat_completion(
                    messages=[{"role": "user", "content": body}],
                    max_tokens=5, temperature=0.0)["choices"][0]["message"]["content"]
            L = letter(t)
            tot += 1
            ok += (L == exp)
            bcount += (L == "B")
            single += (len(t.strip()) == 1)
    print(f"{mode}: acc {ok}/{tot} B-rate {bcount/tot:.2f} single-char {single}/{tot}", flush=True)
