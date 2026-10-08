"""Historical WikiText LoRA adapter recovery, with configurable artifact paths."""
import argparse
from pathlib import Path


def main(argv=None, default_scale="s20"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=f"D:/qwen/models/Qwen2.5-0.5B-pruned-{default_scale}")
    parser.add_argument("--out-adapter", default=f"D:/qwen/models/{default_scale}-lora-adapter")
    parser.add_argument("--out-masks", default=f"D:/qwen/models/{default_scale}-masks.pt")
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    if Path(args.src).resolve() == Path(args.out_adapter).resolve():
        parser.error("--out-adapter must differ from --src")

    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.src)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError("tokenizer must define a padding or EOS token")
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.float16).cuda()
    masks = {name: (module.weight.detach() == 0).cpu()
             for name, module in model.named_modules()
             if isinstance(module, nn.Linear) and "layers" in name}
    if not masks:
        raise ValueError("source model has no decoder linear layers to mask")
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"][:args.samples * 2]
             if len(t.strip()) > 100][:args.samples]
    if not texts:
        raise ValueError("no training examples matched the historical selection")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    model.train()
    total = 0.0
    for index in range(0, len(texts), 2):
        batch = tok(texts[index:index + 2], return_tensors="pt", truncation=True,
                    max_length=512, padding=True).to("cuda")
        labels = batch.input_ids.clone()
        labels[batch.attention_mask == 0] = -100
        loss = model(**batch, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at batch {index // 2 + 1}")
        loss.backward()
        opt.step()
        opt.zero_grad()
        total += loss.item()
        step = index // 2 + 1
        if step % 30 == 0:
            print(f"step {step} loss {total / step:.3f}", flush=True)
    model.save_pretrained(args.out_adapter)
    tok.save_pretrained(args.out_adapter)
    mask_path = Path(args.out_masks)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(masks, mask_path)
    print("saved", args.out_adapter, args.out_masks)


if __name__ == "__main__":
    main()
