"""V3.1 unified cross-scale free-response: accepted-alias matching, N reps,
paired transitions, full provenance manifest.
Usage:
  python reproduce/eval_v3b.py --model <hf-dir> --questions data/v2/questions.json
      --out out.json --reps 3 --tag s30-05B
"""
import argparse
import hashlib
import json
import re
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def norm(s):
    return re.sub(r"[\s,]", "", s.lower())


def final_number(text):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return norm(m.group(1))
    m = re.search(r"####\s*(-?[\d,.]+)", text)
    if m:
        return norm(m.group(1))
    mg = re.findall(r"-?\d[\d,.]*", text)
    return norm(mg[-1]) if mg else None


def latex_canon(s):
    s = norm(s)
    s = s.replace("\\pi", "pi").replace("π", "pi")
    for w in ("$", "\\(", "\\)", "\\[", "\\]",
              "<|im_end|>", "<|im_start|>"):
        s = s.replace(w, "")
    return s


def to_number(s):
    s = norm(s)
    try:
        return float(s)
    except Exception:
        pass
    m = re.match(r"^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def match(q, out):
    t = q.get("answer_type", "numeric_scalar")
    if t == "numeric_scalar":
        got = final_number(out)
        if got is None:
            return False
        return any((lambda v: v is not None and abs(v - g) < 1e-9)(
            to_number(a)) for a in q["accepted"]
            if (g := to_number(got)) is not None)
    if t == "unordered_numeric_set":
        nout = latex_canon(out)
        return any(latex_canon(a) in nout for a in q["accepted"])
    if t == "symbolic_exact":
        nout = latex_canon(out).rstrip(".")
        return any(nout == latex_canon(a) or nout.endswith(latex_canon(a))
                   for a in q["accepted"])
    raise ValueError(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=30)
    a = ap.parse_args()

    with open(a.questions, encoding="utf-8", buffering=1) as f:
        qraw = f.read()
    Qs = json.loads(qraw)
    tok = AutoTokenizer.from_pretrained(a.tok or a.model)
    m = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, device_map="cuda",
        low_cpu_mem_usage=True).eval()
    rep_scores, rep_items = [], []
    with torch.no_grad():
        for rep in range(a.reps):
            ok, items = 0, []
            for q in Qs:
                body = tok.apply_chat_template(
                    [{"role": "user",
                      "content": f"{q['question']} Answer with only the value, no explanation."}],
                    tokenize=False, add_generation_prompt=True)
                ids = tok(body, return_tensors="pt").to("cuda")
                out = m.generate(**ids, max_new_tokens=a.max_tokens,
                                 do_sample=False,
                                 eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.eos_token_id)
                t = tok.decode(out[0][ids.input_ids.shape[1]:])
                good = match(q, t)
                ok += good
                items.append({"id": q["id"], "pass": good, "out": t[:300]})
            rep_scores.append(ok)
            rep_items.append(items)
            print(f"rep {rep + 1}: {ok}/{len(Qs)}", flush=True)
    res = {"run_id": f"V31-{a.tag}-{time.strftime('%Y%m%d-%H%M')}",
           "model": a.model,
           "protocol": "v3.1-final-answer-accepted-alias",
           "script_sha256": hashlib.sha256(
               open(__file__, "rb").read()).hexdigest()[:12],
           "dataset": a.questions,
           "dataset_sha256": hashlib.sha256(qraw.encode()).hexdigest()[:12],
           "torch": torch.__version__, "transformers": transformers.__version__,
           "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0),
           "max_new_tokens": a.max_tokens, "do_sample": False,
           "rep_scores": rep_scores, "rep_items": rep_items}
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"saved {a.out}: {rep_scores}")


main()
