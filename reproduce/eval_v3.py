"""V3 unified cross-scale free-response: 1 script/prompt/parser cho s0/s30 x 3 scales.
Parser: final-answer numeric-exact (khong substring), + alias cho symbolic.
Usage:
  python reproduce/eval_v3.py --model <hf-dir> --questions data/v2/questions.json --out out.json
"""
import argparse
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ALIASES = {
    "4pi": ["4pi", "4π", "4*pi", "4 pi"],
    "3x^2": ["3x^2", "3x²", "3*x^2"],
}


def norm(s):
    return re.sub(r"[\s,]", "", s.lower())


def final_number(text):
    """So cuoi cung trong output (uu tien boxed/####), None neu khong co."""
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return norm(m.group(1))
    m = re.search(r"####\s*(-?[\d,.]+)", text)
    if m:
        return norm(m.group(1))
    mg = re.findall(r"-?\d[\d,.]*", text)
    return norm(mg[-1]) if mg else None


def is_numeric(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def match(exp, out):
    """True neu dap an cuoi khop expected (numeric-exact hoac alias)."""
    expn = norm(exp)
    if expn in ALIASES:
        return any(a in norm(out) for a in ALIASES[expn])
    if is_numeric(expn):
        got = final_number(out)
        if got is None or not is_numeric(got):
            return False
        return abs(float(got) - float(expn)) < 1e-9
    return bool(re.search(r"(?<![\w.])" + re.escape(expn) + r"(?![\w.])",
                          norm(out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=30)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tok or a.model)
    m = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, device_map="cuda",
        low_cpu_mem_usage=True).eval()
    Qs = json.load(open(a.questions, encoding="utf-8"))
    ok, raws = 0, []
    with torch.no_grad():
        for q in Qs:
            opts, ans = q["choices"], q["answer"]
            exp = opts["ABCD".index(ans)]
            body = tok.apply_chat_template(
                [{"role": "user", "content": f"{q['question']} Answer with only the value, no explanation."}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(body, return_tensors="pt").to("cuda")
            out = m.generate(**ids, max_new_tokens=a.max_tokens, do_sample=False,
                             eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id)
            t = tok.decode(out[0][ids.input_ids.shape[1]:])
            good = match(exp, t)
            ok += good
            raws.append({"id": q["id"], "exp": exp, "pass": good, "out": t[:300]})
    res = {"model": a.model, "protocol": "v3-final-answer", "free": ok,
           "total": len(Qs), "items": raws,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M")}
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"V3 free {ok}/{len(Qs)}")


main()
