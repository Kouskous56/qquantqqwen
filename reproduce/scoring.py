"""Canonical V3.2 answer matcher (single source of truth).
Typed semantics: numeric_scalar / unordered_numeric_set /
ordered_numeric_tuple / symbolic_exact.
"""
import math
import re


NUMBER_PATTERN = r"-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?)?"


def norm(s):
    return re.sub(r"[\s,]", "", s.lower())


def latex_canon(s):
    s = norm(s)
    s = s.replace("\\pi", "pi").replace("π", "pi")
    for w in ("$", "\\(", "\\)", "\\[", "\\]", "<|im_end|>", "<|im_start|>"):
        s = s.replace(w, "")
    return s


def final_number(text):
    """Return (normalized value, clean_flag). clean=False when the matched
    number is glued to trailing letters (e.g. '0.5x')."""
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return norm(m.group(1)), True
    m = re.search(r"####\s*(" + NUMBER_PATTERN + r")", text)
    if m:
        glued = m.end() < len(text) and text[m.end()].isalpha()
        return norm(m.group(1)), not glued
    mg = list(re.finditer(NUMBER_PATTERN, text))
    if not mg:
        return None, False
    last = mg[-1]
    glued = last.end() < len(text) and text[last.end()].isalpha()
    return norm(last.group(0)), not glued


def to_number(s):
    s = norm(s)
    try:
        value = float(s)
        return value if math.isfinite(value) else None
    except ValueError:
        pass
    m = re.match(r"^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$", s)
    if m and float(m.group(2)) != 0:
        value = float(m.group(1)) / float(m.group(2))
        return value if math.isfinite(value) else None
    return None


def num_set(text):
    """Parse ALL numbers in text into a sorted tuple, or None if none."""
    values = num_seq(text)
    return tuple(sorted(values)) if values is not None else None


def num_seq(text):
    """Parse ALL numbers in text preserving order, or None if none."""
    nums = []
    for m in re.finditer(NUMBER_PATTERN, text):
        v = to_number(m.group())
        # Invalid numbers must not vanish and turn a longer answer into a match.
        if v is None or (m.end() < len(text) and text[m.end()].isalpha()):
            return None
        nums.append(v)
    return tuple(nums) if nums else None


def match(q, out):
    t = q.get("answer_type", "numeric_scalar")
    if t == "numeric_scalar":
        got, clean = final_number(out)
        if got is None or not clean:
            return False
        for a in q["accepted"]:
            v, g = to_number(a), to_number(got)
            if v is not None and g is not None and abs(v - g) < 1e-9:
                return True
        return False
    if t == "unordered_numeric_set":
        got = num_set(out)
        if got is None:
            return False
        for a in q["accepted"]:
            av = num_set(a)
            if av is not None and len(av) == len(got) and all(abs(x - y) < 1e-9
                                           for x, y in zip(av, got)):
                return True
        return False
    if t == "ordered_numeric_tuple":
        got = num_seq(out)
        if got is None:
            return False
        for a in q["accepted"]:
            av = num_seq(a)
            if av is not None and len(av) == len(got) and all(abs(x - y) < 1e-9
                                           for x, y in zip(av, got)):
                return True
        return False
    if t == "symbolic_exact":
        nout = latex_canon(out).rstrip(".")
        return any(nout == latex_canon(a) for a in q["accepted"])
    raise ValueError(t)
