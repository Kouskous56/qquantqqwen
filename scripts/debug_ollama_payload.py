from lm_eval.models.openai_completions import LocalCompletionsAPI
import requests
m = LocalCompletionsAPI(
    model="qwen3b-mine-q4",
    base_url="http://127.0.0.1:11434/v1/completions",
    num_concurrent=1, max_retries=1, tokenized_requests=False,
    tokenizer="D:/qwen/models/Qwen2.5-3B-FP16",
    eos_string="<|im_end|>")
p = m._create_payload(
    "Write a limerick.",
    generate=True,
    gen_kwargs={"until": [], "max_gen_toks": 1280},
    seed=1234, eos=m.eos_string)
print("PAYLOAD KEYS:", list(p.keys()))
print("prompt type:", type(p["prompt"]).__name__, "| len:", len(p["prompt"]))
print("max_tokens:", repr(p["max_tokens"]), "| temp:", repr(p["temperature"]))
print("stop:", p["stop"], "| seed:", repr(p["seed"]))
r = requests.post("http://127.0.0.1:11434/v1/completions", json=p)
print("status:", r.status_code)
print(r.text[:300])
