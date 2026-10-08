"""Compare paired GSM blocks by question ID, without rerunning inference.

Reject incomplete, overlapping or incompatible blocks. Stored pass labels are
audited against exp/got; this does not validate the original answer extractor.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load_blocks(paths):
    records, sources, model = {}, [], None
    for value in paths:
        path = Path(value)
        raw = path.read_bytes()
        run = json.loads(raw)
        if model is not None and model != run.get("model"):
            raise ValueError("Blocks within an arm must use the same model")
        model = run.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{path.name}: missing model")
        details = run.get("details", [])
        if not details or len(details) != run.get("n"):
            raise ValueError(f"{path.name}: incomplete item count")
        local_ids = []
        for item in details:
            index = item["i"]
            if type(index) is not int or index < 0 or index in records:
                raise ValueError(f"{path.name}: invalid/duplicate item ID {index}")
            if type(item.get("pass")) is not bool:
                raise ValueError(f"{path.name}: pass label must be boolean")
            if not isinstance(item.get("exp"), str) or not isinstance(item.get("got"), str):
                raise ValueError(f"{path.name}: missing expected/extracted answer")
            if item["pass"] != (item["exp"] == item["got"]):
                raise ValueError(f"{path.name}: inconsistent pass label for {index}")
            records[index] = item
            local_ids.append(index)
        if "offset" in run and sorted(local_ids) != list(range(run["offset"], run["offset"] + run["n"])):
            raise ValueError(f"{path.name}: item range does not match offset/n")
        if "gsm" in run and run["gsm"] != sum(item["pass"] for item in details):
            raise ValueError(f"{path.name}: aggregate score differs from details")
        sources.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
    if not records:
        raise ValueError("At least one complete block is required per arm")
    return model, records, sources


def compare(arm_a, arm_b):
    model_a, a, sources_a = load_blocks(arm_a)
    model_b, b, sources_b = load_blocks(arm_b)
    if a.keys() != b.keys():
        raise ValueError("Arms must contain exactly the same question IDs")
    if any(a[i]["exp"] != b[i]["exp"] for i in a):
        raise ValueError("Expected answers differ between arms")
    n11 = sum(a[i]["pass"] and b[i]["pass"] for i in a)
    n10 = sum(a[i]["pass"] and not b[i]["pass"] for i in a)
    n01 = sum(not a[i]["pass"] and b[i]["pass"] for i in a)
    n00 = len(a) - n11 - n10 - n01
    discordant = n10 + n01
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(n10, n01) + 1)) / 2**discordant)
    return {"model_a": model_a, "model_b": model_b, "n": len(a),
            "index_range": [min(a), max(a)], "a_correct": n11 + n10,
            "b_correct": n11 + n01, "a_minus_b_pp": 100 * (n10 - n01) / len(a),
            "n11": n11, "n10": n10, "n01": n01, "n00": n00,
            "mcnemar_exact_two_sided_p": p,
            "sources_a": sources_a, "sources_b": sources_b,
            "limitation": "Recounts stored labels; matching IDs/answers alone cannot establish identical generation protocols or dataset question text."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", nargs="+", required=True, type=Path)
    parser.add_argument("--arm-b", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.resolve() in {p.resolve() for p in args.arm_a + args.arm_b}:
        parser.error("Output must not overwrite an input run")
    result = compare(args.arm_a, args.arm_b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("sources_")}, indent=2))


if __name__ == "__main__":
    main()
