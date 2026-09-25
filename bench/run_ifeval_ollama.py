"""IFEval qua Ollama bang Python API (bool that, tranh CLI string bug)."""
import lm_eval
res = lm_eval.simple_evaluate(
    model="local-completions",
    model_args={
        "model": "qwen3b-mine-q4",
        "base_url": "http://127.0.0.1:11434/v1/completions",
        "num_concurrent": 1,
        "max_retries": 3,
        "tokenized_requests": False,
        "tokenizer": "D:/qwen/models/Qwen2.5-3B-FP16",
        "eos_string": "<|im_end|>",
    },
    tasks=["ifeval"],
    batch_size=1,
    log_samples=True,
    verbosity="INFO",
)
print("RESULTS:", res["results"])
import json
json.dump({k: (str(v)[:500]) for k, v in res["results"].items()},
          open("D:/qwen/notes/lmeval/ifeval_mineq4_summary.json", "w"))
json.dump(res.get("samples", {}),
          open("D:/qwen/notes/lmeval/ifeval_mineq4_samples.json", "w"),
          ensure_ascii=False, default=str)
print("saved samples")
