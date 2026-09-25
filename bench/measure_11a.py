"""11A: realized systems benefit - dense FP16 vs s30/s40/s50 (transformers, CUDA).
Workload: 3 prompts x 100 tokens. Metrics: load/VRAM/TTFT/prefill/decode/power."""
import torch, time, threading, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage

BASE = "D:/qwen/models/"
MODELS = {"dense": BASE + "Qwen2.5-3B-FP16",
          "s30": BASE + "Qwen2.5-3B-FP16-pruned-s30",
          "s40": BASE + "Qwen2.5-3B-FP16-pruned-s40",
          "s50": BASE + "Qwen2.5-3B-FP16-pruned-s50"}
tok = AutoTokenizer.from_pretrained(BASE + "Qwen2.5-3B-FP16")
PROMPTS = ["Explain photosynthesis in two sentences.",
           "Write a Python function to sort a list.",
           "What is 37 times 24? Explain briefly."]
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)

class PowerMon:
    def __init__(self):
        self.ws, self.on = [], True
    def run(self):
        while self.on:
            try:
                self.ws.append(nvmlDeviceGetPowerUsage(h) / 1000.0)
            except Exception:
                pass
            time.sleep(0.1)
    def avg(self):
        return sum(self.ws) / max(len(self.ws), 1)

class FirstTok(TextStreamer):
    def __init__(self, t):
        super().__init__(t, skip_prompt=True)
        self.t0 = None
    def on_finalized_text(self, text, stream_end=False):
        if self.t0 is None:
            self.t0 = time.time()
        super().on_finalized_text(text, stream_end)

R = {}
for name, path in MODELS.items():
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                             device_map="cuda",
                                             low_cpu_mem_usage=True).eval()
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    tot_tok, tot_dec, tot_pre, tot_wall, ttfts, walls = 0, 0.0, 0.0, 0.0, [], []
    mon = PowerMon()
    th = threading.Thread(target=mon.run, daemon=True)
    th.start()
    with torch.no_grad():
        for p in PROMPTS:
            body = tok.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(body, return_tensors="pt").to("cuda")
            t1 = time.time()
            _ = m(ids.input_ids)
            pre_s = time.time() - t1
            st = FirstTok(tok)
            t2 = time.time()
            out = m.generate(**ids, max_new_tokens=100, do_sample=False,
                             eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id, streamer=st)
            wall = time.time() - t2
            ntok = out.shape[1] - ids.input_ids.shape[1]
            ttfts.append(st.t0 - t2 if st.t0 else -1)
            tot_tok += ntok
            tot_dec += wall
            tot_pre += pre_s
            tot_wall += wall
            walls.append(round(wall, 2))
    mon.on = False
    th.join()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    avgw = mon.avg()
    R[name] = {"load_s": round(load_s, 1), "peak_vram_gb": round(peak, 2),
               "ttft_s": [round(x, 3) for x in ttfts],
               "prefill_tps": round(sum(len(tok(p)["input_ids"]) for p in PROMPTS) / tot_pre, 1),
               "decode_tps": round(tot_tok / tot_dec, 1),
               "wall_s": round(tot_wall, 1), "avg_power_w": round(avgw, 1),
               "joule_per_token": round(avgw * tot_wall / max(tot_tok, 1), 3)}
    print(f"{name}: {json.dumps(R[name])}", flush=True)
    del m
    torch.cuda.empty_cache()
json.dump(R, open("D:/qwen/notes/RUN_11A.json", "w"), indent=1)
print("saved RUN_11A.json")
