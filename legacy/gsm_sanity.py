"""Sanity: manual template vs apply_chat_template tren 1.5B s0, 3 cau GSM de."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
P = "D:/qwen/models/Qwen2.5-1.5B-FP16"
tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16, device_map="cuda").eval()
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(3))

@torch.no_grad()
def gen(body, mx=200):
    ids = tok(body, return_tensors="pt").to("cuda")
    out = m.generate(**ids, max_new_tokens=mx, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:])

for i, r in enumerate(ds):
    q = r["question"]
    manual = (f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
    ids = tok.apply_chat_template([{"role": "user", "content": q}],
                                  tokenize=False, add_generation_prompt=True)
    print(f"===== Q{i} manual =====")
    print(gen(manual)[:400])
    print(f"===== Q{i} apply_chat_template =====")
    print(gen(ids)[:400])
    print(f"----- expected: {r['answer'][-20:]}")
