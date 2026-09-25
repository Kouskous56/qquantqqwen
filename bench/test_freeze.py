"""Task 9A: freeze harness + repeatability. Free-response 50Q x (F16,Q4) x 3 reps,
cung script, cung params. Log manifest day du."""
from llama_cpp import Llama
import json, time, re, hashlib, platform
import torch, transformers, llama_cpp

BASE = "D:/qwen/models/Qwen2.5-0.5B/"
MODELS = {"F16": BASE + "my-qwen05-f16.gguf", "Q4": BASE + "my-qwen05-q4_k_m.gguf"}
PARAMS = {"n_ctx": 1024, "max_tokens": 30, "temperature": 0.0}
src = open("D:/qwen/bench/test50_math.py", encoding="utf-8").read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = [(cid, q, re.findall(r'"(.*?)"', opts), "ABCD".index(ans))
      for cid, _, q, opts, ans in pat.findall(src)]
dhash = hashlib.sha256(src.encode()).hexdigest()[:12]

manifest = {
    "run_id": "9A-" + time.strftime("%Y%m%d-%H%M"),
    "params": PARAMS,
    "dataset_hash": dhash,
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                 "python": platform.python_version()},
    "models": {n: p for n, p in MODELS.items()},
    "note": "chat-template via create_chat_completion, greedy",
}
print("manifest:", json.dumps(manifest), flush=True)

def norm(s):
    return re.sub(r"[\s,]", "", s.lower())

llms = {n: Llama(model_path=p, n_ctx=PARAMS["n_ctx"], verbose=False) for n, p in MODELS.items()}
rep_scores = {n: [] for n in MODELS}
for rep in range(3):
    for n, llm in llms.items():
        ok = 0
        for cid, q, opts, cidx in Qs:
            t = llm.create_chat_completion(
                messages=[{"role": "user", "content": f"{q} Answer with only the value, no explanation:"}],
                max_tokens=PARAMS["max_tokens"], temperature=PARAMS["temperature"]
            )["choices"][0]["message"]["content"]
            expv = norm(opts[cidx])
            ok += (bool(expv) and expv in norm(t))
        rep_scores[n].append(ok)
        print(f"rep {rep+1} {n}: free {ok}/50", flush=True)

manifest["rep_scores"] = rep_scores
json.dump(manifest, open("D:/qwen/notes/RUN_9A.json", "w"), indent=1)
print("saved RUN_9A.json")
