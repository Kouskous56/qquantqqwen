"""11B1 eval: SparseGPT-2:4 3B - PPL + V2 free + perm + GSM (frozen harness)."""
import torch, torch.nn as nn, re, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import Counter
P = "D:/qwen/models/Qwen2.5-3B-sparsegpt-24"
tok = AutoTokenizer.from_pretrained("D:/qwen/models/Qwen2.5-3B-FP16")
wikitxt = open("D:/qwen/bench/wikitext_test.txt", encoding="utf-8").read()
src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(cid, q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for cid, _, q, opts, ans in pat.findall(src)]

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())

def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

def chat(q):
    return tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)

def extract(t):
    m_ = re.search(r"\\boxed\{([^}]+)\}", t)
    if m_:
        return m_.group(1).replace(",", "").strip()
    m_ = re.search(r"####\s*(-?[\d,.]+)", t)
    if m_:
        return m_.group(1).replace(",", "")
    mg = re.findall(r"-?\d[\d,.]*", t)
    return mg[-1].replace(",", "") if mg else ""

@torch.no_grad()
def gen(m, body, mx):
    ids = tok(body, return_tensors="pt").to("cuda")
    out = m.generate(**ids, max_new_tokens=mx, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:])

m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16, device_map="cuda",
                                         low_cpu_mem_usage=True).eval()
enc = tok(wikitxt, return_tensors="pt")
seq, stride, mx = enc.input_ids.size(1), 256, 512
nlls, nt = [], 0
m.eval()
with torch.no_grad():
    for i in range(0, seq, stride):
        b = max(i + stride - mx, 0)
        e = min(i + stride, seq)
        ids = enc.input_ids[:, b:e].cuda()
        tgt = ids.clone()
        tgt[:, :-stride] = -100
        out = m(ids, labels=tgt)
        nlls.append(out.loss * (e - max(b, e - stride)))
        nt += e - max(b, e - stride)
pp = torch.exp(torch.stack(nlls).sum() / nt).item()
print(f"sgpt24 PPL {pp:.2f}", flush=True)
free = 0
for cid, q, opts, cidx in Qs:
    t = gen(m, chat(f"{q} Answer with only the value, no explanation. "
                    "Think step by step, then end with: #### <number>"), 60)
    expv = norm(opts[cidx])
    free += (bool(expv) and expv in norm(t))
print(f"sgpt24 free {free}/50", flush=True)
letters, perq = [], {}
for cid, q, opts, cidx in Qs:
    for r in range(4):
        order = [opts[(i + r) % 4] for i in range(4)]
        exp = "ABCD"[(cidx - r) % 4]
        L = letter(gen(m, chat(f"Question: {q}\nA) {order[0]}\nB) {order[1]}\nC) {order[2]}\nD) {order[3]}\nAnswer with only the letter A, B, C or D:"), 5))
        letters.append(L)
        perq.setdefault(cid, {})[r] = (L == exp)
    print(f"perm done {cid}", flush=True)
c = Counter(letters)
perm = sum(1 for cid in perq for r in range(4) if perq[cid][r])
rot = {r: sum(1 for cid in perq if perq[cid][r]) for r in range(4)}
cons = sum(1 for cid in perq if all(perq[cid][r] for r in range(4)))
print(f"sgpt24 perm {perm}/200 Br {c['B']/200:.2f} cons {cons} rot {rot} dist {dict(c)}", flush=True)
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(50))
g = 0
for rr in ds:
    t = gen(m, chat(rr["question"] + "\nThink step by step, then end with: #### <number>"), 512)
    me = re.search(r"####\s*(-?[\d,.]+)", rr["answer"])
    exp = me.group(1).replace(",", "") if me else ""
    g += (extract(t) == exp)
print(f"sgpt24 gsm {g}/50", flush=True)
json.dump({"sgpt24": {"ppl": round(pp, 2), "free": free, "perm": perm, "gsm": g,
                      "b_rate": round(c["B"] / 200, 3), "cons4": cons, "rot": rot}},
          open("D:/qwen/notes/RUN_11B1.json", "w"), indent=1)
print("saved RUN_11B1.json")
