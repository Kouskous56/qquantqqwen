"""Sequential SparseGPT adapter for Qwen2; copy alongside the upstream package.

Capture the model's causal masks and rotary embeddings before pruning. This
keeps calibration consistent with the installed Transformers implementation.
"""
import argparse
import math
import time

import torch
import torch.nn as nn

DEV = torch.device("cuda:0")


def get_qwen(model_path):
    from transformers import Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, low_cpu_mem_usage=True)
    model.seqlen = min(2048, model.config.max_position_embeddings)
    return model


def find_layers(module, layers=(nn.Linear,), name=""):
    if isinstance(module, layers):
        return {name: module}
    result = {}
    for child_name, child in module.named_children():
        result.update(find_layers(child, layers, f"{name}.{child_name}" if name else child_name))
    return result


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _layer_output(layer, hidden, kwargs, device):
    output = layer(hidden_states=hidden.unsqueeze(0), **_to_device(kwargs, device))
    # Transformers 4 returns a tuple; Transformers 5 returns the tensor directly.
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.squeeze(0)


class _CalibrationCaptured(Exception):
    """Internal control flow; genuine model ValueErrors must propagate."""


@torch.no_grad()
def qwen_sequential(model, dataloader, dev, options, sparsegpt_cls=None):
    if options.nsamples < 1:
        raise ValueError("nsamples must be positive")
    if len(set(getattr(model.config, "layer_types", []) or [])) > 1:
        raise ValueError("mixed sliding/full attention needs per-layer masks; unsupported by this adapter")
    if sparsegpt_cls is None:
        from sparsegpt import SparseGPT
        sparsegpt_cls = SparseGPT
    layers = model.model.layers
    use_cache = model.config.use_cache
    model.config.use_cache = False
    first_layer = layers[0]
    dtype = next(model.parameters()).dtype
    inps = torch.empty((options.nsamples, model.seqlen, model.config.hidden_size),
                       dtype=dtype, device=dev)
    captured_kwargs = []
    embedding_modules = ["embed_tokens", "norm"]
    if hasattr(model.model, "rotary_emb"):
        embedding_modules.append("rotary_emb")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if tuple(inp.shape) != (1, model.seqlen, model.config.hidden_size):
                raise ValueError("calibration requires one full-length sequence per batch")
            inps[len(captured_kwargs)].copy_(inp[0])
            captured_kwargs.append(_to_device(kwargs, "cpu"))
            raise _CalibrationCaptured

    try:
        for name in embedding_modules:
            getattr(model.model, name).to(dev)
        first_layer.to(dev)
        layers[0] = Catcher(first_layer)
        try:
            for batch in dataloader:
                if len(captured_kwargs) == options.nsamples:
                    break
                try:
                    model(batch[0].to(dev), use_cache=False)
                except _CalibrationCaptured:
                    pass
        finally:
            layers[0] = first_layer.cpu()
            for name in embedding_modules:
                getattr(model.model, name).cpu()
        if len(captured_kwargs) != options.nsamples:
            raise ValueError(f"captured {len(captured_kwargs)} calibration samples; expected {options.nsamples}")
        outs = torch.empty_like(inps)
        for index, layer in enumerate(layers):
            layer.to(dev)
            subset = find_layers(layer)
            gpts = {name: sparsegpt_cls(module) for name, module in subset.items()}
            handles = []
            try:
                def add_batch(name):
                    def hook(_, inputs, output):
                        gpts[name].add_batch(inputs[0].detach(), output.detach())
                    return hook
                for name, module in subset.items():
                    handles.append(module.register_forward_hook(add_batch(name)))
                try:
                    for sample, kwargs in enumerate(captured_kwargs):
                        outs[sample].copy_(_layer_output(layer, inps[sample], kwargs, dev))
                finally:
                    for handle in handles:
                        handle.remove()
                for name, gpt in gpts.items():
                    print(index, name, flush=True)
                    gpt.fasterprune(options.sparsity, prunen=options.prunen,
                                    prunem=options.prunem, percdamp=options.percdamp,
                                    blocksize=options.blocksize)
                for sample, kwargs in enumerate(captured_kwargs):
                    outs[sample].copy_(_layer_output(layer, inps[sample], kwargs, dev))
            finally:
                for handle in handles:
                    handle.remove()
                for gpt in gpts.values():
                    gpt.free()
                layer.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            inps, outs = outs, inps
    finally:
        model.config.use_cache = use_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("dataset", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--save", default="")
    args = parser.parse_args(argv)
    if args.nsamples < 1 or args.blocksize < 1:
        parser.error("--nsamples and --blocksize must be positive")
    if not math.isfinite(args.sparsity) or not 0 <= args.sparsity <= 1:
        parser.error("--sparsity must be between 0 and 1")
    if not math.isfinite(args.percdamp) or args.percdamp <= 0:
        parser.error("--percdamp must be positive")
    if not (args.prunen == args.prunem == 0 or 0 < args.prunen < args.prunem):
        parser.error("structured pruning requires 0 < --prunen < --prunem")
    from datautils import get_loaders
    model = get_qwen(args.model).eval()
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples,
                              seed=args.seed, model=args.model, seqlen=model.seqlen)
    started = time.time()
    qwen_sequential(model, dataloader, DEV, args)
    print("prune time:", round(time.time() - started, 1))
    if args.save:
        from transformers import AutoTokenizer
        model.save_pretrained(args.save)
        AutoTokenizer.from_pretrained(args.model).save_pretrained(args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
