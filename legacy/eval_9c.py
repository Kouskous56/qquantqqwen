"""9C eval: merge thuong vs masked merge. Do sparsity + PPL."""
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SRC = "D:/qwen/models/Qwen2.5-0.5B-pruned-s20"
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

base = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float16).cuda()
print(f"s20 base: sparsity {sparsity(base):.3f}", flush=True)
pm = PeftModel.from_pretrained(base, "D:/qwen/models/s20-lora-adapter")
merged = pm.merge_and_unload()
sp_plain = sparsity(merged)
print(f"merged-plain: sparsity {sp_plain:.4f}", flush=True)
merged.save_pretrained("D:/qwen/models/s20-lora-merged")
print(f"PPL plain {ppl(merged):.2f}", flush=True)
masks = torch.load("D:/qwen/models/s20-masks.pt")
with torch.no_grad():
    for n, m in merged.named_modules():
        if isinstance(m, nn.Linear) and n in masks:
            m.weight.data *= (~masks[n].cuda())
print(f"merged-masked: sparsity {sparsity(merged):.3f}", flush=True)
merged.save_pretrained("D:/qwen/models/s20-lora-masked")
print(f"PPL masked {ppl(merged):.2f}", flush=True)
