"""Demo chatbot Task 8: 3B mine-Q4 chinh + 0.5B dense-Q4 baseline. 8 prompts."""
import json, time, subprocess, urllib.request

API = "http://127.0.0.1:11434/api/generate"
OLLAMA = r"C:\Users\ACER\AppData\Local\Programs\Ollama\ollama.exe"
MODELS = ["qwen3b-mine-q4", "qwen2.5:0.5b"]
PROMPTS = [
    ("factual-en", "What is the capital of Japan? Answer in one sentence."),
    ("vietnamese", "Giai thich quang hop trong 2 cau."),
    ("arithmetic", "What is 37 times 24? Show the steps briefly."),
    ("reasoning", "If all roses are flowers and some flowers fade quickly, do all roses fade quickly? Explain in one sentence."),
    ("code", "Write a Python function to check if a number is prime. Only code."),
    ("json", 'Reply with only this JSON and nothing else: {"name": "Hanoi", "country": "Vietnam"}'),
    ("long-explain", "Explain how neural networks learn, in about 100 words."),
    ("collapse-trigger", "The capital of France is"),
]

def gen(model, prompt):
    req = urllib.request.Request(API, data=json.dumps(
        {"model": model, "prompt": prompt, "stream": False,
         "options": {"num_ctx": 4096, "num_predict": 200}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    wall = time.time() - t0
    return d, wall

def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    return o + " MiB"

L = ["# Demo chatbot Task 8", ""]
for m in MODELS:
    subprocess.run([OLLAMA, "stop", m], capture_output=True)
for m in MODELS:
    L.append(f"## Model: {m}")
    for pid, p in PROMPTS:
        d, wall = gen(m, p)
        toks = d.get("eval_count", 0)
        tps = toks / d["eval_duration"] * 1e9 if d.get("eval_duration") else 0
        ttft = d.get("prompt_eval_duration", 0) / 1e9
        out = d.get("response", "").strip().replace("\n", " ")
        line = f"[{pid}] wall {wall:.1f}s TTFT {ttft:.2f}s tok/s {tps:.1f} vram {vram()}"
        print(f"{m} {line}", flush=True)
        L.append(f"### {pid} ({line})\nQ: {p}\nA: {out[:800]}\n")
    subprocess.run([OLLAMA, "stop", m], capture_output=True)
open("D:/qwen/notes/DEMO_CHATBOT.md", "w", encoding="utf-8").write("\n".join(L))
print("saved DEMO_CHATBOT.md")
