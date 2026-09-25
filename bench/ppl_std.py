"""Standardized PPL: 9 checkpoints, 1 evaluator (512/256, token-weighted, no template)."""
import torch, json, time, hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer

B = "D:/qwen/models/"
MODELS = [
    ("05B-s0", B + "Qwen2.5-0.5B-FP16", B + "Qwen2.5-0.5B-FP16"),
    ("05B-s30", B + "Qwen2.5-0.5B-pruned-s30", B + "Qwen2.5-0.5B-FP16"),
    ("15B-s0", B + "Qwen2.5-1.5B-FP16", B + "Qwen2.5-1.5B-FP16"),
    ("15B-s30", B + "Qwen2.5-1.5B-FP16-pruned-s30", B + "Qwen2.5-1.5B-FP16"),
    ("3B-s0", B + "Qwen2.5-3B-FP16", B + "Qwen2.5-3B-FP16"),
    ("3B-s30", B + "Qwen2.5-3B-FP16-pruned-s30", B + "Qwen2.5-3B-FP16"),
    ("3B-s40", B + "Qwen2.5-3B-FP16-pruned-s40", B + "Qwen2.5-3B-FP16"),
    ("3B-s50", B + "Qwen2.5-3B-FP16-pruned-s50", B + "Qwen2.5-3B-FP16"),
    ("3B-sgpt24", B + "Qwen2.5-3B-sparsegpt-24", B + "Qwen2.5-3B-FP16"),
]
text = open("D:/qwen/bench/wikitext_test.txt", encoding="utf-8").read()
print("corpus sha:", hashlib.sha256(text.encode()).hexdigest()[:12], flush=True)
R = {}
for name, mp, tp in MODELS:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(tp)
    m = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.float16,
                                             device_map="cuda",
                                             low_cpu_mem_usage=True).eval()
    enc = tok(text, return_tensors="pt")
    seq, stride, mx = enc.input_ids.size(1), 256, 512
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
    pp = torch.exp(torch.stack(nlls).sum() / nt).item()
    R[name] = {"ppl": round(pp, 2), "tokens": nt,
               "runtime_s": round(time.time() - t0, 1)}
    print(f"{name}: PPL {pp:.2f} ({nt} tok, {R[name]['runtime_s']}s)", flush=True)
    del m, tok
    torch.cuda.empty_cache()
json.dump({"config": "512/256 token-weighted no-template fp16",
           "results": R},
          open("D:/qwen/notes/RUN_PPL_STD.json", "w"), indent=1)
print("saved RUN_PPL_STD.json")
