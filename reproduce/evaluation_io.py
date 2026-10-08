"""Small, dependency-free input/output helpers for evaluation entry points."""
import argparse
import json
from pathlib import Path


ANSWER_TYPES = {
    "numeric_scalar", "unordered_numeric_set", "ordered_numeric_tuple",
    "symbolic_exact",
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def validate_questions(questions, *, typed=False):
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a non-empty list")
    seen = set()
    for q in questions:
        if not isinstance(q, dict) or not isinstance(q.get("id"), str) or not q["id"]:
            raise ValueError("each question must have a non-empty string id")
        qid = q["id"]
        if qid in seen:
            raise ValueError(f"duplicate question id: {qid}")
        seen.add(qid)
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            raise ValueError(f"{qid}: question must be non-empty text")
        if typed:
            if q.get("answer_type", "numeric_scalar") not in ANSWER_TYPES:
                raise ValueError(f"{qid}: unsupported answer_type")
            accepted = q.get("accepted")
            if not isinstance(accepted, list) or not accepted or not all(
                isinstance(a, str) and a.strip() for a in accepted
            ):
                raise ValueError(f"{qid}: accepted must contain non-empty strings")
        else:
            choices = q.get("choices")
            if not isinstance(choices, list) or len(choices) != 4 or not all(
                isinstance(c, str) and c.strip() for c in choices
            ):
                raise ValueError(f"{qid}: exactly four non-empty choices are required")
            if q.get("answer") not in ("A", "B", "C", "D"):
                raise ValueError(f"{qid}: answer must be A, B, C or D")
    return questions


def load_questions(path, *, typed=False):
    with open(path, encoding="utf-8") as f:
        return validate_questions(json.load(f), typed=typed)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
