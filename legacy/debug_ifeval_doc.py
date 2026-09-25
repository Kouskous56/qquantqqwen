from lm_eval.tasks import TaskManager
from lm_eval.models.openai_completions import LocalCompletionsAPI
import requests
_tm = TaskManager()
_loaded = _tm.load_task_or_group("ifeval")
task = _loaded["ifeval"] if isinstance(_loaded, dict) else _loaded
docs = list(task.fewshot_docs() if hasattr(task, "fewshot_docs") else [])
print("has fewshot_docs:", bool(docs))
d0 = task.task_docs[0] if hasattr(task, "task_docs") else None
print("keys:", list(d0.keys()) if isinstance(d0, dict) else type(d0))
inst = d0.get("instruction") or d0.get("prompt") or str(d0)[:200]
print("INST:", str(inst)[:300])
m = LocalCompletionsAPI(
    model="qwen3b-mine-q4",
    base_url="http://127.0.0.1:11434/v1/completions",
    num_concurrent=1, max_retries=1, tokenized_requests=False,
    tokenizer="D:/qwen/models/Qwen2.5-3B-FP16",
    eos_string="<|im_end|>")
p = m._create_payload(str(inst), generate=True,
                      gen_kwargs={"until": [], "max_gen_toks": 1280},
                      seed=1234, eos=m.eos_string)
r = requests.post("http://127.0.0.1:11434/v1/completions", json=p)
print("status:", r.status_code)
print(r.text[:400])
