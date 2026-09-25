"""GSM audit 5 cau: full output + extract check."""
import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
P = "D:/qwen/models/Qwen2.5-1.5B-FP16"
tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16, device_map="cuda").eval()
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(5))

@torch.no_grad()
def gen(body, mx=512):
    ids = tok(body, return_tensors="pt").to("cuda")
    out = m.generate(**ids, max_new_tokens=mx, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:])

for i, r in enumerate(ds):
    q = r["question"] + "\nThink step by step, then end with: #### <number>"
    body = tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)
    t = gen(body, 512)
    me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
    exp = me.group(1).replace(",", "") if me else ""
    mg = re.findall(r"-?[\d,.]+", t)
    got = mg[-1].replace(",", "") if mg else ""
    print(f"===== Q{i} exp={exp} got={got} match={got==exp} =====")
    print(t[-500:])
