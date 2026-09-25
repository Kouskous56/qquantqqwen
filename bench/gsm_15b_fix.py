"""10D-fix: GSM 1.5B s0/s20/s30 voi extractor dung (boxed/####/last-number)."""
import torch, re, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE = "D:/qwen/models/"
MODELS = {"s0": BASE + "Qwen2.5-1.5B-FP16",
          "s20": BASE + "Qwen2.5-1.5B-FP16-pruned-s20",
          "s30": BASE + "Qwen2.5-1.5B-FP16-pruned-s30"}
only = sys.argv[1:] or list(MODELS)
MODELS = {k: v for k, v in MODELS.items() if k in only}
tok = AutoTokenizer.from_pretrained(BASE + "Qwen2.5-1.5B-FP16")
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(50))

def extract(t):
    m = re.search(r"\\boxed\{([^}]+)\}", t)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"####\s*(-?[\d,.]+)", t)
    if m:
        return m.group(1).replace(",", "")
    mg = re.findall(r"-?\d[\d,.]*", t)
    return mg[-1].replace(",", "") if mg else ""

@torch.no_grad()
def gen(m, body, mx=512):
    ids = tok(body, return_tensors="pt").to("cuda")
    out = m.generate(**ids, max_new_tokens=mx, do_sample=False,
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:])

for name, path in MODELS.items():
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda").eval()
    ok, raws = 0, []
    for r in ds:
        q = r["question"] + "\nThink step by step, then end with: #### <number>"
        body = tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
        t = gen(m, body)
        me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
        exp = me.group(1).replace(",", "") if me else ""
        got = extract(t)
        ok += (got == exp)
        raws.append({"exp": exp, "got": got})
    print(f"{name}: gsm-fixed {ok}/50", flush=True)
    import json
    json.dump(raws, open(f"D:/qwen/notes/GSM15B_{name}.json", "w"), indent=1)
    del m
    torch.cuda.empty_cache()
print("saved")
