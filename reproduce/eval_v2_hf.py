"""Historical V2 evaluator, HF backend (same-harness family as 10B).
Keeps the historical ChatML prompt and permissive letter/substring scoring.
Use eval_v3b.py for canonical typed free-response scores.
Usage:
  python reproduce/eval_v2_hf.py --model <hf-dir> --questions data/v2/questions.json --out out.json
"""
import argparse
import re
import time
from collections import Counter

if __package__:
    from .evaluation_io import load_questions, positive_int, write_json
else:
    from evaluation_io import load_questions, positive_int, write_json


def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"


def norm(s):
    return re.sub(r"[\s,]", "", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=positive_int, default=30)
    a = ap.parse_args()

    Qs = load_questions(a.questions)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tok or a.model)
    m = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, device_map="cuda",
        low_cpu_mem_usage=True).eval()

    def ask(body, mx):
        prompt = (f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n")
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=mx, do_sample=False,
                             eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.input_ids.shape[1]:])

    letters, perq, free, praw = [], {}, 0, {}
    for q in Qs:
        opts, cidx = q["choices"], "ABCD".index(q["answer"])
        for r in range(4):
            order = [opts[(i + r) % 4] for i in range(4)]
            exp = "ABCD"[(cidx - r) % 4]
            p = (f"Question: {q['question']}\nA) {order[0]}\nB) {order[1]}\n"
                 f"C) {order[2]}\nD) {order[3]}\n"
                 "Answer with only the letter A, B, C or D:")
            t = ask(p, 5)
            got = letter(t)
            letters.append(got)
            perq.setdefault(q["id"], {})[r] = {"got": got, "exp": exp,
                                               "out": t}
        fp = f"{q['question']} Answer with only the value, no explanation:"
        expv = norm(opts[cidx])
        tf = ask(fp, a.max_tokens)
        free += (bool(expv) and expv in norm(tf))
        praw[q["id"]] = tf
    c = Counter(letters)
    ok = lambda cid, r: perq[cid][r]["got"] == perq[cid][r]["exp"]
    perm = sum(1 for cid in perq for r in range(4) if ok(cid, r))
    cons = sum(1 for cid in perq if all(ok(cid, r) for r in range(4)))
    res = {"model": a.model, "backend": "hf-transformers-greedy-chat-template",
           "protocol": "v2-historical-letter-substring",
           "tokenizer": a.tok or a.model,
           "perm": perm, "perm_total": len(letters),
           "b_rate": round(c["B"] / len(letters), 3),
           "consistent_4of4": cons, "free": free, "free_total": len(Qs),
           "dist": dict(c), "per_question": perq, "free_raw": praw,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M")}
    write_json(a.out, res)
    print(f"perm {perm}/{len(letters)} cons {cons} free {free}/{len(Qs)} B-rate {res['b_rate']}")


if __name__ == "__main__":
    main()
