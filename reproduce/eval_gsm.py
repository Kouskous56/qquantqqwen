"""GSM8K held-out evaluator (boxed -> #### -> number).
Historical string scoring remains the default for the published comparison.
Use --scoring numeric for exact numeric equality and the last answer marker;
the selected scoring protocol is recorded in every new output.
Usage:
  python reproduce/eval_gsm.py --model <hf-dir> --n 50 --out out.json
"""
import argparse
import re
import time
from fractions import Fraction

if __package__:
    from .evaluation_io import positive_int, write_json
else:
    from evaluation_io import positive_int, write_json


NUMERIC_PATTERN = (
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|(?=\.\d))"
    r"(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"(?:/[+-]?\d+(?:\.\d+)?)?"
)


def extract(t):
    """Frozen historical extractor; do not change published scoring silently."""
    m = re.search(r"\\boxed\{([^}]+)\}", t)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"####\s*(-?[\d,.]+)", t)
    if m:
        return m.group(1).replace(",", "")
    mg = re.findall(r"-?\d[\d,.]*", t)
    return mg[-1].replace(",", "") if mg else ""


def extract_numeric(text):
    """Extract a finite number, honoring the last boxed/#### answer.

    Fraction comparison avoids float rounding and accepts equivalent decimal,
    integer, thousands-separated and rational spellings. Invalid explicit
    answers are rejected rather than falling back to an earlier calculation.
    """
    text = text.replace("−", "-")
    boxed = list(re.finditer(r"\\boxed\{([^}]+)\}", text))
    if boxed:
        candidate = boxed[-1].group(1).strip()
        if not re.fullmatch(NUMERIC_PATTERN, candidate):
            return ""
    else:
        marked = list(re.finditer(r"####\s*", text))
        if marked:
            answer = text[marked[-1].end():]
            number = re.match(NUMERIC_PATTERN, answer)
        else:
            answer = text
            numbers = list(re.finditer(NUMERIC_PATTERN, answer))
            number = numbers[-1] if numbers else None
        if number is None:
            return ""
        # Do not mistake a suffix of an identifier or malformed numeric token
        # for a scalar; sentence-ending punctuation remains allowed.
        before = answer[number.start() - 1:number.start()] if number.start() else ""
        after = answer[number.end():number.end() + 1]
        if (before and (before.isalnum() or before in "_./")) or (
            after and (after.isalnum() or after in "_/," )
        ):
            return ""
        candidate = number.group()
    try:
        pieces = candidate.replace(",", "").split("/")
        value = Fraction(pieces[0])
        if len(pieces) == 2:
            value /= Fraction(pieces[1])
        return str(value)
    except (ValueError, ZeroDivisionError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--n", type=positive_int, default=50)
    ap.add_argument("--max-tokens", type=positive_int, default=512)
    ap.add_argument("--scoring", choices=("historical", "numeric"), default="historical")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from datasets import load_dataset

    test = load_dataset("openai/gsm8k", "main", split="test")
    if a.n > len(test):
        ap.error(f"--n must not exceed the test split size ({len(test)})")
    ds = test.select(range(a.n))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tok or a.model)
    m = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, device_map="cuda",
        low_cpu_mem_usage=True).eval()
    ok, raws = 0, []
    with torch.no_grad():
        for index, r in enumerate(ds):
            body = tok.apply_chat_template(
                [{"role": "user",
                  "content": r["question"] + "\nThink step by step, then end with: #### <number>"}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(body, return_tensors="pt", add_special_tokens=False).to("cuda")
            out = m.generate(**ids, max_new_tokens=a.max_tokens, do_sample=False,
                             eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id)
            t = tok.decode(out[0][ids.input_ids.shape[1]:])
            if a.scoring == "numeric":
                exp, got = extract_numeric(r["answer"]), extract_numeric(t)
            else:
                me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
                exp = me.group(1).replace(",", "") if me else ""
                got = extract(t)
            if not exp:
                raise ValueError(f"GSM8K test item {index} has no valid reference answer")
            good = got == exp
            ok += good
            raws.append({"i": index, "question": r["question"], "exp": exp,
                         "got": got, "pass": good, "out": t})
    res = {"model": a.model, "n": a.n, "correct": ok,
           "tokenizer": a.tok or a.model,
           "protocol": f"gsm8k-{a.scoring}-boxed-hash-number",
           "dataset": "openai/gsm8k", "config": "main", "split": "test",
           "offset": 0, "max_new_tokens": a.max_tokens, "do_sample": False,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M")}
    write_json(a.out, {"summary": res, "items": raws})
    print(f"gsm {ok}/{a.n}")


if __name__ == "__main__":
    main()
