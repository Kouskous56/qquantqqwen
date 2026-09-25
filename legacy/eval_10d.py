"""10D: 1.5B s0/s20/s30 x PPL + V2 free + perm (full dissection) + GSM/50."""
import torch, torch.nn as nn, re, json, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import Counter

BASE = "D:/qwen/models/"
MODELS = {"s0": BASE + "Qwen2.5-1.5B-FP16",
          "s20": BASE + "Qwen2.5-1.5B-FP16-pruned-s20",
          "s30": BASE + "Qwen2.5-1.5B-FP16-pruned-s30"}
TOK_SRC = BASE + "Qwen2.5-1.5B-FP16"
tok = AutoTokenizer.from_pretrained(TOK_SRC)

src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(cid, q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for cid, _, q, opts, ans in pat.findall(src)]
wikitxt = open("D:/qwen/bench/wikitext_test.txt", encoding="utf-8").read()

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())

def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

@torch.no_grad()
def gen(m, body, mx):
    ids = tok(body, return_tensors="pt").to("cuda")
    out = m.generate(**ids, max_new_tokens=mx, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:])

def ppl(m):
    enc = tok(wikitxt, return_tensors="pt")
    seq, stride, mx = enc.input_ids.size(1), 512, 1024
    nlls, nt = [], 0
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
    return torch.exp(torch.stack(nlls).sum() / nt).item()

def gsm(m, n=50):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    ok = 0
    for r in ds:
        t = gen(m, f"<|im_start|>user\n{r['question']}<|im_end|>\n<|im_start|>assistant\n", 200)
        me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
        exp = me.group(1).replace(",", "") if me else ""
        mg = re.findall(r"-?[\d,.]+", t)
        got = mg[-1].replace(",", "") if mg else ""
        ok += (got == exp)
    return ok

R = {}
for name, path in MODELS.items():
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda").eval()
    pp = ppl(m)
    free = perm_ok = 0
    letters, perq = [], {}
    for cid, q, opts, cidx in Qs:
        t = gen(m, f"<|im_start|>user\n{q} Answer with only the value, no explanation.<|im_end|>\n<|im_start|>assistant\n", 30)
        expv = norm(opts[cidx])
        free += (bool(expv) and expv in norm(t))
        for r in range(4):
            order = [opts[(i + r) % 4] for i in range(4)]
            exp = "ABCD"[(cidx - r) % 4]
            p = (f"<|im_start|>user\nQuestion: {q}\nA) {order[0]}\nB) {order[1]}\nC) {order[2]}\nD) {order[3]}\n"
                 f"Answer with only the letter A, B, C or D:<|im_end|>\n<|im_start|>assistant\n")
            L = letter(gen(m, p, 5))
            letters.append(L)
            perq.setdefault(cid, {})[r] = (L == exp)
    c = Counter(letters)
    perm_ok = sum(1 for cid in perq for r in range(4) if perq[cid][r])
    rot = {r: sum(1 for cid in perq if perq[cid][r]) for r in range(4)}
    cons = sum(1 for cid in perq if all(perq[cid][r] for r in range(4)))
    g = gsm(m)
    R[name] = {"ppl": round(pp, 2), "free": free, "perm": perm_ok, "gsm": g,
               "dist": dict(c), "rot": rot, "cons4": cons}
    print(f"{name}: PPL {pp:.2f} free {free}/50 perm {perm_ok}/200 gsm {g}/50 "
          f"Br {c['B']/200:.2f} cons {cons} rot {rot}", flush=True)
    del m
    torch.cuda.empty_cache()
json.dump({"run_id": "10D-" + time.strftime("%Y%m%d-%H%M"), "results": R},
          open("D:/qwen/notes/RUN_10D_15B.json", "w"), indent=1)
print("saved RUN_10D_15B.json")
