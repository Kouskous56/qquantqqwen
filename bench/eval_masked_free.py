"""Free-response 1 rep cho masked model (transformers, template tay)."""
import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("D:/qwen/models/Qwen2.5-0.5B-pruned-s20")
m = AutoModelForCausalLM.from_pretrained(
    "D:/qwen/models/s20-lora-masked", dtype=torch.float16, device_map="cuda").eval()
src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for _, _, q, opts, ans in pat.findall(src)]

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())

ok = 0
for q, opts, cidx in Qs:
    body = (f"<|im_start|>user\n{q} Answer with only the value, no explanation."
            f"<|im_end|>\n<|im_start|>assistant\n")
    ids = tok(body, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=30, do_sample=False,
                         eos_token_id=tok.eos_token_id,
                         pad_token_id=tok.eos_token_id)
    t = tok.decode(out[0][ids.input_ids.shape[1]:])
    expv = norm(opts[cidx])
    ok += (bool(expv) and expv in norm(t))
print(f"masked free-response: {ok}/50")
