import time, requests, json
URL = "http://127.0.0.1:11434/api/generate"
PROMPTS = [
    "What is 2+2? Answer short.",
    "Write a python function to add two numbers.",
    "Giai thich quantize INT8 trong 2 cau.",
]
for model in ["qwen2.5:0.5b"]:
    for p in PROMPTS:
        t0 = time.time()
        r = requests.post(URL, json={"model": model, "prompt": p, "stream": False}, timeout=120)
        dt = time.time() - t0
        txt = r.json().get("response", "")[:300]
        print(f"[{model}] {dt:.1f}s | {p[:40]} -> {txt[:120]}")
