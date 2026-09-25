"""10B eval: s30 masked merge + PPL + free-response (frozen)."""
import torch, torch.nn as nn, re
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def ppl(model):
    text = open("D:/qwen/bench/wikitext_test.txt", encoding="utf-8").read()
    enc = tok(text, return_tensors="pt")
    seq, stride, mx = enc.input_ids.size(1), 512, 1024
    nlls, nt = [], 0
    model.eval()
    with torch.no_grad():
        for i in range(0, seq, stride):
            b = max(i + stride - mx, 0)
            e = min(i + stride, seq)
            ids = enc.input_ids[:, b:e].cuda()
            tgt = ids.clone()
            tgt[:, :-stride] = -100
            out = model(ids, labels=tgt)
            nlls.append(out.loss * (e - max(b, e - stride)))
            nt += e - max(b, e - stride)
    return torch.exp(torch.stack(nlls).sum() / nt).item()

def free_resp(model):
    src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
    pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
    Qs = [(q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
          for _, _, q, opts, ans in pat.findall(src)]
    def norm(s):
        return re.sub(r"[\s,]", "", s.lower())
    ok = 0
    for q, opts, cidx in Qs:
        body = (f"<|im_start|>user\n{q} Answer with only the value, no explanation."
                f"<|im_end|>\n<|im_start|>assistant\n")
        ids = tok(body, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=30, do_sample=False,
                                 eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.eos_token_id)
        t = tok.decode(out[0][ids.input_ids.shape[1]:])
        expv = norm(opts[cidx])
        ok += (bool(expv) and expv in norm(t))
    return ok

base = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float16).cuda()
print(f"s30 base: sparsity {sparsity(base):.3f} PPL {ppl(base):.2f} free {free_resp(base)}/50", flush=True)
pm = PeftModel.from_pretrained(base, "D:/qwen/models/s30-lora-adapter")
merged = pm.merge_and_unload()
print(f"plain: sparsity {sparsity(merged):.4f} PPL {ppl(merged):.2f} free {free_resp(merged)}/50", flush=True)
merged.save_pretrained("D:/qwen/models/s30-lora-merged")
masks = torch.load("D:/qwen/models/s30-masks.pt")
with torch.no_grad():
    for n, m in merged.named_modules():
        if isinstance(m, nn.Linear) and n in masks:
            m.weight.data *= (~masks[n].cuda())
print(f"masked: sparsity {sparsity(merged):.3f} PPL {ppl(merged):.2f} free {free_resp(merged)}/50", flush=True)
merged.save_pretrained("D:/qwen/models/s30-lora-masked")
print("saved s30-lora-masked")
