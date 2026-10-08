"""Canonical Wanda pruning for Qwen2 (unstructured per-output + 2:4).
Usage:
  python reproduce/prune_wanda.py --src <hf-dir> --out <dir> --sparsity 0.3 [--semi24]
"""
import argparse
import math

import torch


def get_linears(layer):
    attn, mlp = layer.self_attn, layer.mlp
    return {"q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj,
            "o": attn.o_proj, "gate": mlp.gate_proj, "up": mlp.up_proj,
            "down": mlp.down_proj}


def prune_weight(weight, scaler, sparsity, semi24=False):
    """Prune exactly floor(width * sparsity) entries per row, including ties."""
    if not math.isfinite(sparsity) or not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be between 0 and 1")
    if weight.ndim != 2 or scaler.shape != (weight.shape[1],):
        raise ValueError("expected a matrix and one activation scale per input column")
    metric = weight.detach().float().abs() * scaler.float().unsqueeze(0)
    if not torch.isfinite(metric).all():
        raise ValueError("pruning metric contains non-finite values")
    if semi24:
        if weight.shape[1] % 4:
            raise ValueError("2:4 pruning requires an input width divisible by four")
        grouped = metric.reshape(weight.shape[0], -1, 4)
        indices = torch.argsort(grouped, dim=-1, stable=True)[..., :2]
        mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, indices, True)
        mask = mask.reshape(weight.shape)
    else:
        count = int(weight.shape[1] * sparsity)
        indices = torch.argsort(metric, dim=1, stable=True)[:, :count]
        mask = torch.zeros_like(metric, dtype=torch.bool).scatter_(1, indices, True)
    return weight.detach().masked_fill(mask, 0)


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sparsity", type=float, default=0.3)
    ap.add_argument("--semi24", action="store_true")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--calib", default="Salesforce/wikitext")
    a = ap.parse_args(argv)
    if not math.isfinite(a.sparsity) or not 0 <= a.sparsity <= 1:
        ap.error("--sparsity must be between 0 and 1")
    if a.nsamples < 1 or a.seqlen < 2:
        ap.error("--nsamples must be positive and --seqlen must be at least 2")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.src)
    model = AutoModelForCausalLM.from_pretrained(
        a.src, dtype=torch.float16, device_map="cuda").eval()
    layers = model.model.layers

    class Rec:
        def __init__(self):
            self.sum2, self.cnt = None, 0

        def __call__(self, mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            s = x.pow(2).sum(dim=0)
            self.sum2 = s if self.sum2 is None else self.sum2 + s
            self.cnt += x.shape[0]

    ds = load_dataset(a.calib, "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"][:a.nsamples * 3]
             if len(t.strip()) > 100][:a.nsamples]
    if len(texts) < a.nsamples:
        raise ValueError(
            f"only {len(texts)} calibration texts for {a.nsamples} requested; "
            "3B-s40/s50 historical runs used --nsamples 8 --seqlen 128 under VRAM pressure")
    cal = [tok(t, return_tensors="pt", truncation=True,
               max_length=a.seqlen).input_ids.cuda() for t in texts]

    for li, layer in enumerate(layers):
        linears = get_linears(layer)
        recs = {n: Rec() for n in linears}
        hooks = [l.register_forward_hook(recs[n]) for n, l in linears.items()]
        try:
            for ids in cal:
                model(ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for n, l in linears.items():
            if recs[n].cnt == 0:
                raise RuntimeError(f"no calibration activations captured for layer {li} {n}")
            scaler = (recs[n].sum2 / recs[n].cnt).sqrt()
            l.weight.copy_(prune_weight(l.weight, scaler, a.sparsity, a.semi24))
        print(f"layer {li + 1}/{len(layers)}", flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
