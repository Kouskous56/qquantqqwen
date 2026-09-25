"""Canonical V2 evaluator: 4-way option permutation + free-response.
Usage:
  python reproduce/eval_v2.py --model <gguf> --questions data/v2/questions.json --out out.json
"""
import argparse
import json
import re
import time
from collections import Counter

from llama_cpp import Llama


def letter(t):
    m = re.search(r"[ABCD]", t.upper())
    return m.group(0) if m else "?"


def norm(s):
    return re.sub(r"[\s,]", "", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--use-template", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="wrap prompts with the model chat template (raw mode is legacy)")
    a = ap.parse_args()

    Qs = json.load(open(a.questions, encoding="utf-8"))
    llm = Llama(model_path=a.model, n_ctx=a.n_ctx, verbose=False)

    def ask(body, mx):
        if a.use_template:
            return llm.create_chat_completion(
                messages=[{"role": "user", "content": body}],
                max_tokens=mx, temperature=0.0)["choices"][0]["message"]["content"]
        return llm(body, max_tokens=mx, temperature=0.0)["choices"][0]["text"]

    letters, perq, free, raws = [], {}, 0, {}
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
                                               "out": t[:200]}
        fp = f"{q['question']} Answer with only the value, no explanation:"
        expv = norm(opts[cidx])
        tf = ask(fp, a.max_tokens)
        free += (bool(expv) and expv in norm(tf))
        raws[q["id"]] = {"free_out": tf[:200]}
    c = Counter(letters)
    ok = lambda cid, r: perq[cid][r]["got"] == perq[cid][r]["exp"]
    perm = sum(1 for cid in perq for r in range(4) if ok(cid, r))
    cons = sum(1 for cid in perq if all(ok(cid, r) for r in range(4)))
    res = {"model": a.model, "use_template": a.use_template,
           "perm": perm, "perm_total": len(letters),
           "b_rate": round(c["B"] / len(letters), 3),
           "consistent_4of4": cons, "free": free,
           "free_total": len(Qs), "dist": dict(c),
           "per_question": perq, "free_raw": raws,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M")}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"perm {perm}/{len(letters)} cons {cons} free {free}/{len(Qs)} B-rate {res['b_rate']}")


main()
