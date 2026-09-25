"""9C: LoRA recovery tren s20-pruned 0.5B. Luu masks truoc train de so masked merge."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

SRC = "D:/qwen/models/Qwen2.5-0.5B-pruned-s30"
tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float16).cuda()
masks = {}
for n, m in model.named_modules():
    import torch.nn as nn
    if isinstance(m, nn.Linear) and "layers" in n:
        masks[n] = (m.weight.data == 0)
print("mask tensors:", len(masks), flush=True)

cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in ds["text"][:600] if len(t.strip()) > 100][:300]
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()
for ep in range(1):
    tot, n = 0.0, 0
    for i in range(0, len(texts), 2):
        batch = tok(texts[i:i + 2], return_tensors="pt", truncation=True,
                    max_length=512, padding=True).to("cuda")
        labels = batch.input_ids.clone()
        labels[batch.attention_mask == 0] = -100
        loss = model(**batch, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        tot += loss.item()
        n += 1
        if n % 30 == 0:
            print(f"step {n} loss {tot/n:.3f}", flush=True)
model.save_pretrained("D:/qwen/models/s30-lora-adapter")
torch.save({k: v.cpu() for k, v in masks.items()},
           "D:/qwen/models/s30-masks.pt")
print("saved adapter + masks")
