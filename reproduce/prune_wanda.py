"""Canonical Wanda pruning for Qwen2 (unstructured per-output + 2:4).
Usage:
  python reproduce/prune_wanda.py --src <hf-dir> --out <dir> --sparsity 0.3 [--semi24]
"""
import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_linears(layer):
    attn, mlp = layer.self_attn, layer.mlp
    return {"q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj,
            "o": attn.o_proj, "gate": mlp.gate_proj, "up": mlp.up_proj,
            "down": mlp.down_proj}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sparsity", type=float, default=0.3)
    ap.add_argument("--semi24", action="store_true")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--calib", default="Salesforce/wikitext")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.src)
    model = AutoModelForCausalLM.from_pretrained(
        a.src, dtype=torch.float16, device_map="cuda").eval()
    layers = model.model.layers

    class Rec:
        def __init__(self):
            self.sum2, self.cnt = None, 0

        def __call__(self, mod, inp, out):
            x = inp[0].detach().float()
            s = x.pow(2).sum(dim=(0, 1))
            self.sum2 = s if self.sum2 is None else self.sum2 + s
            self.cnt += x.shape[0] * x.shape[1]

    ds = load_dataset(a.calib, "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"][:a.nsamples * 3]
             if len(t.strip()) > 100][:a.nsamples]
    assert len(texts) >= a.nsamples, (
        f"only {len(texts)} calibration texts; "
        "3B-s40/s50 historical runs used --nsamples 8 --seqlen 128 under VRAM pressure")
    cal = [tok(t, return_tensors="pt", truncation=True,
               max_length=a.seqlen).input_ids.cuda() for t in texts]

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
            if a.semi24:
                assert W.shape[1] % 4 == 0
                g = metric.view(-1, W.shape[1] // 4, 4)
                _, idx = torch.topk(g, k=2, dim=-1, largest=False)
                mask = torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False)
                Wp = W * mask.view(W.shape).to(W.dtype)
            else:
                k = int(W.shape[1] * a.sparsity)
                if k > 0:
                    thresh = torch.topk(metric, k=k, dim=1,
                                        largest=False)[0][:, -1].unsqueeze(1)
                    Wp = torch.where(metric <= thresh, torch.zeros_like(W), W)
                else:
                    Wp = W
            l.weight.data = Wp.to(torch.float16)
        print(f"layer {li + 1}/{len(layers)}", flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out)


main()
