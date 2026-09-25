from llama_cpp import Llama
import json, time
BASE = "D:/qwen/models/Qwen2.5-0.5B/"
MODELS = {"F16": BASE+"my-qwen05-f16.gguf", "Q8": BASE+"my-qwen05-q8_0.gguf",
          "Q4": BASE+"my-qwen05-q4_k_m.gguf", "Q3": BASE+"my-qwen05-q3_k_m.gguf"}
CASES = [
    ("fact-en-1", "The capital of France is", ["paris"]),
    ("fact-en-2", "Water boils at", ["100"]),
    ("fact-vi-1", "Thu do cua Viet Nam la", ["ha noi", "hanoi"]),
    ("fact-vi-2", "1 nam co bao nhieu ngay", ["365"]),
    ("math-1", "What is 15 times 4? Answer with just the number.", ["60"]),
    ("math-2", "What is 100 minus 37? Answer with just the number.", ["63"]),
    ("reason-1", "If all birds can fly and a penguin is a bird, can a penguin fly? Answer yes or no.", ["no"]),
    ("reason-2", "Tom has 3 apples. He gives 1 away. How many left? Answer with just the number.", ["2"]),
    ("code-py", "Write a python function named add that returns a+b. Only code.", ["def add", "return"]),
    ("code-js", "Write a javascript function named hi that returns 1. Only code.", ["function hi", "return"]),
    ("code-sql", "Write SQL to select all rows from users. Only code.", ["select", "users"]),
    ("translate", "Translate to French: Hello, how are you?", ["bonjour"]),
    ("summarize", "Summarize in 5 words: The cat sat on the mat and slept.", ["cat"]),
    ("json-1", 'Reply with only JSON: {"a": 1}. Text:', ["\"a\"", "1"]),
    ("instruct-1", "Say exactly the word BLUE and nothing else.", ["blue"]),
    ("instruct-2", "List 3 colors separated by commas.", [","]),
    ("vi-qa", " photosynthesis la gi? Tra loi 1 cau.".replace(" photosynthesis", "Quang hop"), ["cay", "anh sang", "nang"]),
    ("creative", "Write a 6-word story about the sea.", ["sea", "ocean", "bien"]),
    ("logic-2", "What comes next: 2, 4, 8, 16, ?", ["32"]),
    ("negation", "Which planet is known as the Red Planet?", ["mars"]),
]
llms = {n: Llama(model_path=p, n_ctx=512, verbose=False) for n, p in MODELS.items()}
rows = []
for cid, prompt, keys in CASES:
    row = {"case": cid, "prompt": prompt}
    for n, llm in llms.items():
        t0 = time.time()
        txt = llm(prompt, max_tokens=60, temperature=0.0)["choices"][0]["text"]
        low = txt.lower()
        ok = any(k in low for k in keys)
        words = low.split()
        repeat = len(words) > 10 and len(set(words)) / len(words) < 0.35
        row[n] = {"pass": ok and not repeat, "repeat": repeat,
                  "s": round(time.time()-t0, 1), "out": txt.replace("\n", " ")[:160]}
    rows.append(row)
    print(f"done {cid}", flush=True)
score = {n: sum(1 for r in rows if r[n]["pass"]) for n in MODELS}
print("SCORE /20:", score)
json.dump(rows, open("D:/qwen/notes/TEST20.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
md = ["# 20 testcases F16 vs quant (0.5B)", "", f"**Score: {score}**", ""]
for r in rows:
    md.append(f"## {r['case']}: {r['prompt']}")
    for n in MODELS:
        d = r[n]
        md.append(f"- {n}: {'PASS' if d['pass'] else 'FAIL'}{' [repeat]' if d['repeat'] else ''} ({d['s']}s) {d['out']}")
    md.append("")
open("D:/qwen/notes/TEST20.md", "w", encoding="utf-8").write("\n".join(md))
print("saved")
