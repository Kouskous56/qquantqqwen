"""SparseGPT cho Qwen2 (adapt tu llama.py upstream): Qwen2ForCausalLM + AutoTokenizer.
VD: python tools/sparsegpt/qwen.py D:/qwen/models/Qwen2.5-3B-FP16 wikitext2
      --nsamples 32 --prunen 2 --prunem 4 --save D:/qwen/models/Qwen2.5-3B-sparsegpt-24
"""
import time
import torch
import torch.nn as nn
from sparsegpt import SparseGPT
from datautils import get_loaders
import argparse

try:
    import wandb as wb
    has_wandb = True
except Exception:
    has_wandb = False

DEV = torch.device("cuda:0")


def get_qwen(model_path):
    from transformers import Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, low_cpu_mem_usage=True)
    model.seqlen = 2048
    return model


def find_layers(module, layers=(nn.Linear,), name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for n, m in module.named_children():
        res.update(find_layers(m, layers=layers,
                               name=name + "." + n if name else n))
    return res


@torch.no_grad()
def qwen_sequential(model, dataloader, dev):
    print("Starting...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size),
                       dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = None
    print("Ready.")

    def rope_cos_sin(seqlen, head_dim, theta, dtype, dev):
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(seqlen, dtype=inv.dtype, device="cpu")
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None].to(dtype).to(dev), emb.sin()[None].to(dtype).to(dev)

    _cos, _sin = rope_cos_sin(
        model.seqlen,
        model.config.hidden_size // model.config.num_attention_heads,
        float(getattr(model.config, "rope_theta", 10000.0)), dtype, dev)
    pos_emb = (_cos, _sin)
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]
        for names in sequential:
            subset = {n: full[n] for n in names}
            gpts = {name: SparseGPT(subset[name]) for name in subset}

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name))
                       for name in subset]
            for j in range(args.nsamples):
                outs[j] = layer(hidden_states=inps[j].unsqueeze(0),
                                attention_mask=attention_mask, position_embeddings=pos_emb)[0]
            for h in handles:
                h.remove()
            for name in subset:
                print(i, name, flush=True)
                gpts[name].fasterprune(args.sparsity, prunen=args.prunen,
                                       prunem=args.prunem, percdamp=args.percdamp,
                                       blocksize=args.blocksize)
                gpts[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(hidden_states=inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=pos_emb)[0]
        layers[i] = layer.cpu()
        del layer, gpts
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    model.config.use_cache = use_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("dataset", type=str, choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    model = get_qwen(args.model)
    model.eval()
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples,
                                seed=args.seed, model=args.model,
                                seqlen=model.seqlen)
    tick = time.time()
    qwen_sequential(model, dataloader, DEV)
    print("prune time:", round(time.time() - tick, 1))
    if args.save:
        model.save_pretrained(args.save)
        print("saved", args.save)
