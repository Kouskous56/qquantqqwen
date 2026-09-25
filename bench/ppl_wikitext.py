import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "D:/qwen/models/Qwen2.5-0.5B-FP16"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda").eval()

text = open("D:/qwen/bench/wikitext_test.txt", encoding="utf-8").read()
enc = tok(text, return_tensors="pt")
seq_len = enc.input_ids.size(1)
stride, max_len = 512, 1024
nlls, n_tokens = [], 0
with torch.no_grad():
    for i in range(0, seq_len, stride):
        begin = max(i + stride - max_len, 0)
        end = min(i + stride, seq_len)
        ids = enc.input_ids[:, begin:end].cuda()
        tgt = ids.clone()
        tgt[:, :-stride] = -100
        out = model(ids, labels=tgt)
        nlls.append(out.loss * (end - max(begin, end - stride)))
        n_tokens += end - max(begin, end - stride)
        if i % 2048 == 0:
            print(f"pos {i}/{seq_len}", flush=True)
ppl = torch.exp(torch.stack(nlls).sum() / n_tokens)
print(f"PPL {MODEL} wikitext-test(200 lines): {ppl.item():.2f}")
open("D:/qwen/notes/PPL_BASELINE.md", "a").write(f"PPL {MODEL}: {ppl.item():.2f}\n")
