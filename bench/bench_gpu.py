import time, json, subprocess, urllib.request

OLLAMA = r"C:\Users\ACER\AppData\Local\Programs\Ollama\ollama.exe"
API = "http://127.0.0.1:11434/api/generate"
PROMPTS = {
    "en": "Explain what a GPU is in two sentences.",
    "vi": "Giai thich GPU trong hai cau.",
    "code": "Write a python function that returns the square of a number. Only code.",
}
MODELS = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen3:4b"]

def post(payload):
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

def vram():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return out + " MiB"

results = []
for m in MODELS:
    subprocess.run([OLLAMA, "stop", m], capture_output=True)
for m in MODELS:
    for pname, prompt in PROMPTS.items():
        r = post({"model": m, "prompt": prompt, "stream": False,
                  "options": {"num_ctx": 4096}})
        eval_s = r["eval_count"] / r["eval_duration"] * 1e9 if r.get("eval_duration") else 0
        prompt_s = r.get("prompt_eval_duration", 0) / 1e9
        total_s = r.get("total_duration", 0) / 1e9
        line = (f"{m} | {pname} | tok/s {eval_s:.1f} | prompt {prompt_s:.2f}s | "
                f"total {total_s:.1f}s | eval_tokens {r.get('eval_count')} | vram {vram()}")
        print(line, flush=True)
        results.append(line)
    subprocess.run([OLLAMA, "stop", m], capture_output=True)

open("D:/qwen/notes/BENCH_BASELINE.md", "w", encoding="utf-8").write(
    "# Bench baseline GPU - " + time.strftime("%d/%m/%Y %H:%M") + "\n\n" +
    "\n".join(results) + "\n")
print("saved notes/BENCH_BASELINE.md")
