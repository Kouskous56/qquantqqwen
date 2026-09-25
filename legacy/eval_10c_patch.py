"""10C patch: GSM-heldout C1 + C2-plain; C2-masked perm distribution."""
import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import Counter

def load(p):
    tok = AutoTokenizer.from_pretrained("D:/qwen/models/Qwen2.5-0.5B-pruned-s30")
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float16, device_map="cuda").eval()
    return tok, m

@torch.no_grad()
def gen(tok, m, body, mx):
    ids = tok(body, return_tensors="pt").to("cuda")
    out = m.generate(**ids, max_new_tokens=mx, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:])

def gsm(tok, m, n=50):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    ok = 0
    for r in ds:
        t = gen(tok, m, f"<|im_start|>user\n{r['question']}<|im_end|>\n<|im_start|>assistant\n", 200)
        me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
        exp = me.group(1).replace(",", "") if me else ""
        mg = re.findall(r"-?[\d,.]+", t)
        got = mg[-1].replace(",", "") if mg else ""
        ok += (got == exp)
    return ok

src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(cid, q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for cid, _, q, opts, ans in pat.findall(src)]

def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

for name, path in [("C1", "D:/qwen/models/s30-lora-masked"),
                   ("C2plain", "D:/qwen/models/s30-gsm8k-merged")]:
    tok, m = load(path)
    print(f"{name}: gsm-heldout {gsm(tok, m)}/50", flush=True)
    del m
    torch.cuda.empty_cache()

tok, m = load("D:/qwen/models/s30-gsm8k-masked")
letters, perot = [], {}
for cid, q, opts, cidx in Qs:
    for r in range(4):
        order = [opts[(i + r) % 4] for i in range(4)]
        exp = "ABCD"[(cidx - r) % 4]
        p = (f"<|im_start|>user\nQuestion: {q}\nA) {order[0]}\nB) {order[1]}\nC) {order[2]}\nD) {order[3]}\n"
             f"Answer with only the letter A, B, C or D:<|im_end|>\n<|im_start|>assistant\n")
        L = letter(gen(tok, m, p, 5))
        letters.append(L)
        perot.setdefault(cid, {})[r] = (L == exp)
c = Counter(letters)
rot = {r: sum(1 for cid in perot if perot[cid][r]) for r in range(4)}
cons = sum(1 for cid in perot if all(perot[cid][r] for r in range(4)))
kdist = {}
for cid in perot:
    k = sum(1 for r in range(4) if perot[cid][r])
    kdist[k] = kdist.get(k, 0) + 1
print(f"C2masked perm: rot {rot} kdist {kdist} cons4 {cons}/50 dist {dict(c)}", flush=True)
