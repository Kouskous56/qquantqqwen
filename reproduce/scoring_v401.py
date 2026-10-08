"""Math50 V4 answer-only scorer. No evaluation of model-produced code.

Numeric equality is exact (Fraction). Symbolic equivalence is a finite,
published alias policy, not a computer algebra system. No last-number fallback.
"""
from fractions import Fraction
import re

VERSION = "math50-scorer-4.0.1"
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?"
TYPES = {"numeric_scalar", "unordered_numeric_multiset", "ordered_numeric_tuple",
         "symbolic_alias", "text_label"}


def unwrap(text):
    if not isinstance(text, str):
        raise ValueError("response must be a string")
    text = text.strip()
    # Only terminal tokens are transport artifacts; embedded tokens stay invalid.
    text = re.sub(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>))+$", "", text).strip()
    prefixed = bool(re.match(r"^(?:final answer\s*:|answer\s*:|####)\s*", text, re.I))
    text = re.sub(r"^(?:final answer\s*:|answer\s*:|####)\s*", "", text, count=1, flags=re.I)
    # Apply at most three wrappers; do not search for an answer inside prose.
    for _ in range(3):
        original = text
        if text.startswith("\\boxed{") and text.endswith("}"):
            inner = text[7:-1]
            depth = 0
            for char in inner:
                depth += (char == "{") - (char == "}")
                if depth < 0:
                    break
            if depth == 0:
                text = inner.strip()
        for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
            if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
                text = text[len(left):-len(right)].strip()
                break
        if text == original:
            break
    return text.replace("−", "-").replace("–", "-"), not prefixed


def number(text, *, grouping=True):
    text = text.strip()
    fraction = re.fullmatch(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", text)
    if fraction:
        text = fraction[1] + "/" + fraction[2]
    parts = re.split(r"\s*/\s*", text)
    if len(parts) > 2 or any(not re.fullmatch(NUMBER, p) for p in parts):
        return None
    if not grouping and "," in text:
        return None
    # Bound adversarial/model-produced exponents and integer sizes.
    if len(text) > 256 or any(abs(int(m)) > 100 for m in re.findall(r"[eE]([+-]?\d+)", text)):
        return None
    try:
        result = Fraction(parts[0].replace(",", ""))
        if len(parts) == 2:
            result /= Fraction(parts[1].replace(",", ""))
        return str(result)
    except (ValueError, ZeroDivisionError):
        return None


def numeric_sequence(text):
    if text and text[0] in "([{":
        pairs = {"(": ")", "[": "]", "{": "}"}
        if not text.endswith(pairs[text[0]]):
            return None
        text = text[1:-1]
    parts = text.split(",")
    if len(parts) < 2:
        return None
    values = [number(part, grouping=False) for part in parts]
    return values if all(value is not None for value in values) else None


def symbolic(text):
    text = text.lower().replace("π", "pi").replace("\\pi", "pi")
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("**", "^")
    text = re.sub(r"\^\{([a-z0-9+-]+)\}", r"^\1", text)
    # Token boundaries and operators must survive normalization: x^2*3 != x^23.
    pattern = r"exp|sqrt|sin|cos|tan|log|ln|pi|[exc]|(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)|[+*/^()=-]"
    tokens = re.findall(pattern, text)
    if ''.join(tokens) != re.sub(r"\s+", "", text) or not tokens:
        return None
    functions = {'exp', 'sqrt', 'sin', 'cos', 'tan', 'log', 'ln'}
    def atom(token):
        return token not in '+*/^()=-'
    result = []
    for token in tokens:
        if result:
            previous = result[-1]
            if (atom(previous) or previous == ')') and (atom(token) or token == '('):
                if previous in functions and token == '(':
                    pass
                elif previous[0].isdigit() and token[0].isdigit():
                    return None
                else:
                    result.append('*')
        result.append(token)
    return ''.join(result)



def normalize(kind, text):
    if kind == "numeric_scalar":
        return number(text)
    if kind in {"unordered_numeric_multiset", "ordered_numeric_tuple"}:
        values = numeric_sequence(text)
        if values is None:
            return None
        return sorted(values, key=Fraction) if kind == "unordered_numeric_multiset" else values
    if kind == "symbolic_alias":
        return symbolic(text)
    if kind == "text_label":
        return " ".join(text.lower().split()).rstrip(".")
    raise ValueError(f"unknown answer type: {kind}")


def score(question, raw, mode="free", expected_letter=None):
    text, compliant = unwrap(raw)
    if mode == "mcq":
        if not isinstance(expected_letter, str) or len(expected_letter) != 1 or expected_letter not in "ABCD":
            raise ValueError("MCQ scoring requires one expected letter")
        normalized = text.upper() if re.fullmatch(r"[A-Da-d]", text) else None
        accepted = [expected_letter]
    elif mode == "free":
        normalized = normalize(question["answer_type"], text)
        accepted = [normalize(question["answer_type"], value) for value in question["accepted"]]
    else:
        raise ValueError("mode must be free or mcq")
    if not text:
        status = "empty"
    elif normalized is None:
        status = "unparseable"
    elif normalized in accepted:
        status = "correct"
    else:
        status = "incorrect"
    return {"pass": status == "correct", "status": status,
            "normalized": normalized, "format_compliant": compliant and bool(text) and normalized is not None,
            "rule": "mcq-letter" if mode == "mcq" else question["answer_type"]}
