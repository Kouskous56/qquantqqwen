"""10C eval: C2 plain/masked - sparsity, V2 free+perm, GSM8K held-out (100 cau)."""
import torch, torch.nn as nn, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

SRC = "D:/qwen/models/Qwen2.5-0.5B-pruned-s30"
tok = AutoTokenizer.from_pretrained(SRC)

def sparsity(model):
    z = t = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "layers" in n:
            z += (m.weight.data == 0).sum().item()
            t += m.weight.data.numel()
    return z / t

src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for _, _, q, opts, ans in pat.findall(src)]

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())

def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"

@torch.no_grad()
def gen(model, body, mx):
    ids = tok(body, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=mx, do_sample=False,
                         eos_token_id=tok.eos_token_id,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:])

def v2(model):
    free = perm = 0
    for q, opts, cidx in Qs:
        t = gen(model, f"<|im_start|>user\n{q} Answer with only the value, no explanation.<|im_end|>\n<|im_start|>assistant\n", 30)
        expv = norm(opts[cidx])
        free += (bool(expv) and expv in norm(t))
        for r in range(4):
            order = [opts[(i + r) % 4] for i in range(4)]
            exp = "ABCD"[(cidx - r) % 4]
            p = (f"<|im_start|>user\nQuestion: {q}\nA) {order[0]}\nB) {order[1]}\nC) {order[2]}\nD) {order[3]}\n"
                 f"Answer with only the letter A, B, C or D:<|im_end|>\n<|im_start|>assistant\n")
            perm += (letter(gen(model, p, 5)) == exp)
    return free, perm

def gsm(model, n=100):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    ok = 0
    for r in ds:
        t = gen(model, f"<|im_start|>user\n{r['question']}<|im_end|>\n<|im_start|>assistant\n", 200)
        m = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
        exp = m.group(1).replace(",", "") if m else ""
        mg = re.findall(r"-?[\d,.]+", t)
        got = mg[-1].replace(",", "") if mg else ""
        ok += (got == exp)
    return ok

base = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float16).cuda()
g0 = gsm(base, 50)
print(f"C0 s30 base: gsm-heldout {g0}/50", flush=True)
pm = PeftModel.from_pretrained(base, "D:/qwen/models/s30-gsm8k-adapter")
merged = pm.merge_and_unload()
sp0 = sparsity(merged)
merged.save_pretrained("D:/qwen/models/s30-gsm8k-merged")
masks = torch.load("D:/qwen/models/s30-masks.pt")
with torch.no_grad():
    for n, m in merged.named_modules():
        if isinstance(m, nn.Linear) and n in masks:
            m.weight.data *= (~masks[n].cuda())
print(f"C2 plain: sparsity {sp0:.4f}", flush=True)
f1, p1 = v2(merged)
print(f"C2 plain(UNMASKED, ref): free {f1}/50 (sparsity da mat, chi de tham khao)", flush=True)
merged2 = AutoModelForCausalLM.from_pretrained("D:/qwen/models/s30-gsm8k-merged", dtype=torch.float16).cuda()
with torch.no_grad():
    for n, m in merged2.named_modules():
        if isinstance(m, nn.Linear) and n in masks:
            m.weight.data *= (~masks[n].cuda())
merged2.save_pretrained("D:/qwen/models/s30-gsm8k-masked")
print(f"C2 masked: sparsity {sparsity(merged2):.3f}", flush=True)
f2, p2 = v2(merged2)
g2 = gsm(merged2, 50)
print(f"C2 masked: free {f2}/50 perm {p2}/200 gsm-heldout {g2}/50", flush=True)
print("saved s30-gsm8k-masked")
