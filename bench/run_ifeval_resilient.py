"""IFEval resilient: Ollama generate truc tiep tung doc (skip-on-error, luu incremental),
roi cham bang harness process_results."""
import json, time, urllib.request, urllib.error, sys
import sys as _s
sys.path.insert(0, "C:/Users/ACER/AppData/Local/Programs/Python/Python311/Lib/site-packages")
from lm_eval.tasks import TaskManager
from lm_eval.tasks.ifeval.utils import process_results

API = "http://127.0.0.1:11434/api/generate"
MODEL = _s.argv[1] if len(_s.argv) > 1 else "qwen3b-mine-q4"
TAG = _s.argv[2] if len(_s.argv) > 2 else "mineq4"
SAVE = f"D:/qwen/notes/lmeval/ifeval_{TAG}_responses.json"

tm = TaskManager()
_loaded = tm.load_task_or_group("ifeval")
task = _loaded["ifeval"] if isinstance(_loaded, dict) else _loaded
docs = list(task.task_docs)
print("docs:", len(docs), flush=True)

# resume
done = {}
try:
    done = {r["key"]: r for r in
            json.load(open(SAVE, encoding="utf-8"))}
    print("resumed:", len(done), flush=True)
except Exception:
    pass

def gen(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"num_ctx": 4096, "num_predict": 1280,
                                   "temperature": 0}}).encode()
    last = "?"
    for _ in range(3):
        try:
            req = urllib.request.Request(API, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)["response"], None
        except Exception as e:
            last = str(e)[:100]
            time.sleep(10)
    return "", last

outs = list(done.values())
for i, d in enumerate(docs):
    key = d.get("key", str(i))
    if key in done:
        continue
    t, err = gen(d["prompt"])
    outs.append({"key": key, "response": t, "error": err})
    if len(outs) % 20 == 0:
        json.dump(outs, open(SAVE, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"saved {len(outs)}/{len(docs)}", flush=True)
json.dump(outs, open(SAVE, "w", encoding="utf-8"), ensure_ascii=False)

# score
ps, pl, i_s, i_l, n_inst = 0, 0, 0, 0, 0
bykey = {d.get("key", str(i)): d for i, d in enumerate(docs)}
for o in outs:
    d = bykey[o["key"]]
    r = process_results(d, [o["response"]])
    a = r.get("prompt_level_strict_acc", 0)
    b = r.get("prompt_level_loose_acc", 0)
    ps += a
    pl += b
    il = r.get("inst_level_strict_acc", [])
    il2 = r.get("inst_level_loose_acc", [])
    i_s += sum(il)
    i_l += sum(il2)
    n_inst += len(il)
n = len(outs)
print(f"prompt-strict {ps/n:.4f} prompt-loose {pl/n:.4f} "
      f"inst-strict {i_s/n_inst:.4f} inst-loose {i_l/n_inst:.4f}")
