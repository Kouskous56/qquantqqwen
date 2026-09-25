from llama_cpp import Llama
import time
FILES = {
    "F16": "D:/qwen/models/Qwen2.5-0.5B/my-qwen05-f16.gguf",
    "Q8_0": "D:/qwen/models/Qwen2.5-0.5B/my-qwen05-q8_0.gguf",
    "Q4_K_M": "D:/qwen/models/Qwen2.5-0.5B/my-qwen05-q4_k_m.gguf",
    "Q3_K_M": "D:/qwen/models/Qwen2.5-0.5B/my-qwen05-q3_k_m.gguf",
}
PROMPT = "The capital of France is"
for name, path in FILES.items():
    t0 = time.time()
    llm = Llama(model_path=path, n_ctx=512, verbose=False)
    out = llm(PROMPT, max_tokens=20, temperature=0.0)
    dt = time.time() - t0
    txt = out["choices"][0]["text"].replace("\n", " ")[:150]
    print(f"[{name}] {dt:.1f}s total (incl load) -> {txt}", flush=True)
    del llm
