"""Wanda pruning cho Qwen2: unstructured per-output + 2:4 structured.
Calibration: wikitext train (32 samples x 512 tokens, nhe cho RAM thap).
VD: python scripts/wanda_qwen.py --sparsity 0.2
    python scripts/wanda_qwen.py --sparsity 0.5 --semi24
"""
import argparse, torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

SRC = "D:/qwen/models/Qwen2.5-0.5B-FP16"

def get_linears(layer):
    attn, mlp = layer.self_attn, layer.mlp
    return {"q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj, "o": attn.o_proj,
            "gate": mlp.gate_proj, "up": mlp.up_proj, "down": mlp.down_proj}

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.2)
    ap.add_argument("--semi24", action="store_true")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--src", type=str, default=SRC)
    a = ap.parse_args()
    src = a.src
    tag = f"s{int(a.sparsity*100)}{'-24' if a.semi24 else ''}"
    print(f"Wanda {src} unstructured={a.sparsity} semi24={a.semi24}", flush=True)

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.float16, device_map="cuda").eval()
    layers = model.model.layers

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"][:a.nsamples * 3] if len(t.strip()) > 100][:a.nsamples]
    cal = [tok(t, return_tensors="pt", truncation=True,
               max_length=a.seqlen).input_ids.cuda() for t in texts]

    class Rec:
        def __init__(self):
            self.sum2, self.cnt = None, 0
        def __call__(self, mod, inp, out):
            x = inp[0].detach().float()
            s = x.pow(2).sum(dim=(0, 1))
            self.sum2 = s if self.sum2 is None else self.sum2 + s
            self.cnt += x.shape[0] * x.shape[1]

    agg = {}
    for li, layer in enumerate(layers):
        linears = get_linears(layer)
        recs = {n: Rec() for n in linears}
        hooks = [l.register_forward_hook(recs[n]) for n, l in linears.items()]
        for ids in cal:
            model(ids, use_cache=False)
        for h in hooks:
            h.remove()
        for n, l in linears.items():
            scaler = (recs[n].sum2 / max(recs[n].cnt, 1)).sqrt().to(torch.float16)
            W = l.weight.data.float()
            metric = W.abs() * scaler.unsqueeze(0)
            Wp = W.clone()
            if a.semi24:
                assert W.shape[1] % 4 == 0
                g = metric.view(-1, W.shape[1] // 4, 4)
                _, idx = torch.topk(g, k=2, dim=-1, largest=False)
                mask = torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False)
                Wp = (W * mask.view(W.shape).to(W.dtype))
            else:
                k = int(W.shape[1] * a.sparsity)
                if k > 0:
                    thresh, _ = torch.topk(metric, k=k, dim=1, largest=False)
                    thresh = thresh[:, -1].unsqueeze(1)
                    Wp = torch.where(metric <= thresh, torch.zeros_like(W), W)
            l.weight.data = Wp.to(torch.float16)
            z = (Wp == 0).sum().item()
            cell = agg.setdefault(n, [0, 0])
            cell[0] += z
            cell[1] += Wp.numel()
        if (li + 1) % 6 == 0 or li + 1 == len(layers):
            print(f"layer {li + 1}/{len(layers)} pruned", flush=True)

    print("per-projection sparsity:")
    for n, (z, t) in agg.items():
        print(f"  {n}: {z/t:.3f}")

    import os
    base = os.path.basename(src.rstrip("/")) + f"-pruned-{tag}"
    out = f"D:/qwen/models/{base}"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print("saved", out)

main()
