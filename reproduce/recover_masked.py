"""Canonical masked recovery: LoRA fine-tune on a pruned model, then reapply mask.
Usage:
  python reproduce/recover_masked.py --src <pruned-hf-dir> --data gsm8k|wikitext
         --out-merged <dir> --out-masked <dir> [--rank 8] [--samples N] [--lr LR]
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]


def mask_prompt_labels(input_ids, prompt_length):
    """Reject examples whose answer was completely removed by truncation."""
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    if not (labels[:, 1:] != -100).any():
        raise ValueError("example has no answer tokens after truncation; shorten the prompt")
    return labels


def restore_masks(model, masks):
    """Fail if merge renamed a layer, instead of silently saving a dense model."""
    modules = dict(model.named_modules())
    missing = set(masks) - set(modules)
    if missing:
        raise ValueError(f"merged model is missing masked layers: {sorted(missing)}")
    with torch.no_grad():
        for name, mask in masks.items():
            weight = modules[name].weight
            if weight.shape != mask.shape:
                raise ValueError(f"mask shape mismatch for {name}")
            weight.masked_fill_(mask.to(weight.device), 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--data", default="gsm8k", choices=["gsm8k", "wikitext"])
    ap.add_argument("--out-merged", required=True)
    ap.add_argument("--out-masked", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed; historical runs set none, so exact-score "
                         "reproduction is not guaranteed for those")
    a = ap.parse_args(argv)
    if a.rank < 1 or a.samples < 1 or not math.isfinite(a.lr) or a.lr <= 0:
        ap.error("--rank, --samples and --lr must be positive")
    paths = [Path(p).resolve() for p in (a.src, a.out_merged, a.out_masked)]
    if len(set(paths)) != 3:
        ap.error("--src, --out-merged and --out-masked must be different directories")

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.src)
    if tok.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if a.seed is not None:
        import random
        import numpy as np
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
    model = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.float16).cuda()
    masks = {}
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "layers" in n:
            masks[n] = (m.weight.detach() == 0).cpu()
    if not masks:
        raise ValueError("source model has no decoder linear layers to mask")

    cfg = LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05,
                     target_modules=TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)

    if a.data == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        items = [{"q": r["question"], "a": r["answer"]} for r in ds][:a.samples]

        def encode(q, ans):
            pre = (f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
            full = pre + ans + "<|im_end|>"
            pe = tok(pre, return_tensors="pt")
            fe = tok(full, return_tensors="pt", truncation=True, max_length=512)
            labels = mask_prompt_labels(fe.input_ids, pe.input_ids.shape[1])
            return fe.input_ids.cuda(), fe.attention_mask.cuda(), labels.cuda()
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"][:a.samples * 2]
                 if len(t.strip()) > 100][:a.samples]

        def encode(t):  # noqa: F811
            e = tok(t, return_tensors="pt", truncation=True,
                    max_length=512, padding=True)
            lb = e.input_ids.clone()
            lb[e.attention_mask == 0] = -100
            return e.input_ids.cuda(), e.attention_mask.cuda(), lb.cuda()
        items = texts

    import torch.nn.functional as F
    if not items:
        raise ValueError("no training examples matched the dataset selection")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr)
    model.train()
    tot, n = 0.0, 0
    for i in range(0, len(items), 2):
        chunk = items[i:i + 2]
        if a.data == "gsm8k":
            ts = [encode(x["q"], x["a"]) for x in chunk]
        else:
            ts = [encode(x) for x in chunk]
        ml = max(t[0].shape[1] for t in ts)
        II = torch.cat([F.pad(t[0], (0, ml - t[0].shape[1]),
                              value=tok.eos_token_id) for t in ts])
        AM = torch.cat([F.pad(t[1], (0, ml - t[1].shape[1])) for t in ts])
        LB = torch.cat([F.pad(t[2], (0, ml - t[2].shape[1]), value=-100) for t in ts])
        loss = model(input_ids=II, attention_mask=AM, labels=LB).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at batch {i // 2 + 1}")
        loss.backward()
        opt.step()
        opt.zero_grad()
        tot += loss.item()
        n += 1
        if n % 30 == 0:
            print(f"step {n} loss {tot / n:.3f}", flush=True)
    merged = model.merge_and_unload()
    tok.save_pretrained(a.out_merged)
    merged.save_pretrained(a.out_merged)
    restore_masks(merged, masks)
    merged.save_pretrained(a.out_masked)
    tok.save_pretrained(a.out_masked)
    print("saved", a.out_merged, a.out_masked)


if __name__ == "__main__":
    main()
