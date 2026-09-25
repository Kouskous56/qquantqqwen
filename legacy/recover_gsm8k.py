"""10C: GSM8K instruction LoRA tren s30. Loss chi tren assistant tokens.
300 samples, batch 2 (=150 steps nhu 10B), rank8, lr 2e-4."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

SRC = "D:/qwen/models/Qwen2.5-0.5B-pruned-s30"
tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float16).cuda()
cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)

ds = load_dataset("openai/gsm8k", "main", split="train")
items = [{"q": r["question"], "a": r["answer"]} for r in ds][:300]

def encode(q, a):
    pre = (f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
    full = pre + a + "<|im_end|>"
    pe = tok(pre, return_tensors="pt")
    fe = tok(full, return_tensors="pt", truncation=True, max_length=512)
    labels = fe.input_ids.clone()
    labels[:, :pe.input_ids.shape[1]] = -100
    return fe.input_ids.cuda(), fe.attention_mask.cuda(), labels.cuda()

opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()
tot, n = 0.0, 0
for i in range(0, len(items), 2):
    bq = items[i:i + 2]
    ids = [encode(x["q"], x["a"]) for x in bq]
    ml = max(t[0].shape[1] for t in ids)
    import torch.nn.functional as F
    II = torch.cat([F.pad(t[0], (0, ml - t[0].shape[1]), value=tok.eos_token_id) for t in ids])
    AM = torch.cat([F.pad(t[1], (0, ml - t[1].shape[1])) for t in ids])
    LB = torch.cat([F.pad(t[2], (0, ml - t[2].shape[1]), value=-100) for t in ids])
    loss = model(input_ids=II, attention_mask=AM, labels=LB).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    tot += loss.item()
    n += 1
    if n % 30 == 0:
        print(f"step {n} loss {tot/n:.3f}", flush=True)
model.save_pretrained("D:/qwen/models/s30-gsm8k-adapter")
print("saved s30-gsm8k-adapter")
