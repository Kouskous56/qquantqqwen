import json
import re


def norm(s):
    return re.sub(r"[\s,]", "", s.lower())


def latex_canon(s):
    s = norm(s)
    s = s.replace("\\pi", "pi").replace("π", "pi")
    for w in ("$", "\\(", "\\)", "\\[", "\\]", "<|im_end|>", "<|im_start|>"):
        s = s.replace(w, "")
    return s


def final_number(text):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return norm(m.group(1))
    m = re.search(r"####\s*(-?[\d,.]+)", text)
    if m:
        return norm(m.group(1))
    mg = re.findall(r"-?\d[\d,.]*", text)
    return norm(mg[-1]) if mg else None


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
    t = q["answer_type"]
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


Qs = {q["id"]: q for q in
      json.load(open("D:/qwen_release/data/v2/questions.json", encoding="utf-8"))}
cases = [
    ("h17", "2, 3<|im_end|>", True),
    ("h17", "3, 2", True),
    ("a48", "ln(e^x) + C<|im_end|>", False),
    ("a48", "e^x + C", True),
    ("h23", "$4\\pi$<|im_end|>", True),
    ("h23", "12.56", False),
    ("h26", "0.5", False),
    ("h26", "0", True),
]
for cid, out, exp in cases:
    got = match(Qs[cid], out)
    flag = "OK" if got == exp else "**MISMATCH**"
    print(f"{flag} {cid} out=[{out[:30]}] -> {got} (expected {exp})")
