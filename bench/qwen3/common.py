"""Shared runtime for new Qwen3 evaluations; historical results stay frozen.

The GSM extractor intentionally retains the published scoring semantics. A
checkpoint is reusable only with identical input data, arguments and sources.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.request

LET = "ABCD"
GSM_PROMPT = "\nThink step by step, then end with: #### <number>"


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return result


def safe_tag(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("tag must contain only letters, digits, _, . or -")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    """Publish complete UTF-8 JSON in one replacement, even after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=1, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def extract(text):
    """Published GSM string matcher (not numeric normalization)."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"####\s*(-?[\d,.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    matches = re.findall(r"-?\d[\d,.]*", text)
    return matches[-1].replace(",", "") if matches else ""


def gold_answer(text):
    match = re.search(r"####\s*(-?[\d,.]+)", text)
    if match is None:
        raise ValueError("GSM dataset answer has no #### numeric gold answer")
    return match.group(1).replace(",", "")


def parse_mc(out, choices):
    text = out.strip()
    match = re.match(r"^[\*\s]*\(?([A-D])\)?(?:[\)\.\:\s\*]|$)", text, re.I)
    if match:
        return LET.index(match.group(1).upper())
    match = re.search(r"answer[^A-D\n]{0,20}([A-D])\b", text, re.I)
    if match:
        return LET.index(match.group(1).upper())
    for i, choice in enumerate(choices):
        if choice and len(choice.strip()) > 2 and choice.strip().lower() in text.lower()[:90]:
            return i
    match = re.search(r"\b([A-D])\b", text)
    return LET.index(match.group(1)) if match else -1


def chat(endpoint, model, content, num_predict):
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": content}],
                       "options": {"num_predict": num_predict, "temperature": 0}}).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    if result.get("error"):
        raise RuntimeError("Ollama: " + str(result["error"]))
    content = result.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Ollama response is missing message.content")
    return content


def run_identity(args, items, task, runner, **extra):
    payload = json.dumps(items, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return {"version": 1, "task": task, "model": args.model,
            "endpoint": args.endpoint.rstrip("/"), "tag": args.tag,
            "dataset_sha256": hashlib.sha256(payload).hexdigest(),
            "indices": [i for i, _ in items],
            "sources": {Path(runner).name: sha256(runner),
                        "common.py": sha256(__file__)}, **extra}


def load_checkpoint(path, identity, items, task):
    try:
        with open(path, encoding="utf-8") as stream:
            checkpoint = json.load(stream)
    except FileNotFoundError:
        return [], 0.0
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid checkpoint {path}; use a new tag or restore the file") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("identity") != identity:
        raise ValueError(f"Checkpoint {path} belongs to a different or legacy run; use a new tag")
    details = checkpoint.get("details")
    if not isinstance(details, list) or len(details) > len(items):
        raise ValueError(f"Invalid checkpoint details: {path}")
    if type(checkpoint.get("done")) is not int or checkpoint["done"] != len(details):
        raise ValueError(f"Invalid checkpoint count: {path}")
    elapsed = checkpoint.get("elapsed_s", 0)
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not 0 <= elapsed < float("inf"):
        raise ValueError(f"Invalid checkpoint elapsed time: {path}")
    for position, row in enumerate(details):
        index, item = items[position]
        if not isinstance(row, dict) or type(row.get("i")) is not int or row["i"] != index:
            raise ValueError(f"Checkpoint indices are not an exact prefix: {path}")
        if type(row.get("pass")) is not bool or not isinstance(row.get("text"), str):
            raise ValueError(f"Checkpoint has invalid score or missing response: {path}")
        if task == "gsm":
            expected, got = gold_answer(item["answer"]), extract(row["text"])
            valid = row.get("exp") == expected and row.get("got") == got and row["pass"] == (got == expected)
        else:
            choices, target = mmlu_choices(item, position)
            pick = parse_mc(row["text"], choices)
            valid = (row.get("subj") == item["subject"] and row.get("gold") == target
                     and row.get("pick") == pick and row["pass"] == (pick == target)
                     and row.get("unparsed") == (pick < 0))
        if not valid:
            raise ValueError(f"Checkpoint score does not match its response and dataset: {path}")
    return details, float(elapsed)


def gsm_items(dataset, offset, count):
    if offset < 0 or count <= 0 or offset + count > len(dataset):
        raise ValueError(f"Requested GSM range [{offset}, {offset + count}) exceeds dataset size {len(dataset)}")
    return [(i, dict(dataset[i])) for i in range(offset, offset + count)]


def mmlu_items(dataset, count=200):
    if count <= 0 or len(dataset) < count:
        raise ValueError(f"MMLU needs {count} items; dataset contains {len(dataset)}")
    by_subject = {}
    for i, row in enumerate(dataset):
        by_subject.setdefault(row["subject"], []).append((i, dict(row)))
    items = []
    for offset in range(max(map(len, by_subject.values()))):
        for subject in sorted(by_subject):
            if offset < len(by_subject[subject]):
                items.append(by_subject[subject][offset])
                if len(items) == count:
                    return items
    raise ValueError("Insufficient MMLU items")


def mmlu_choices(row, position):
    gold, target = row["answer"], position % 4
    choices = row["choices"]
    if len(choices) != 4 or type(gold) is not int or not 0 <= gold < 4:
        raise ValueError("MMLU requires four choices and an integer answer in [0, 3]")
    others = [choice for j, choice in enumerate(choices) if j != gold]
    return others[:target] + [choices[gold]] + others[target:], target


def evaluate(args, items, task, checkpoint, runner, num_predict, **extra):
    identity = run_identity(args, items, task, runner, num_predict=num_predict, **extra)
    details, previous_elapsed = load_checkpoint(checkpoint, identity, items, task)
    started = time.monotonic()
    for position in range(len(details), len(items)):
        index, row = items[position]
        if task == "gsm":
            expected = gold_answer(row["answer"])
            text = chat(args.endpoint, args.model, row["question"] + GSM_PROMPT, num_predict)
            got = extract(text)
            detail = {"i": index, "pass": got == expected, "got": got,
                      "exp": expected, "text": text, "chars": len(text)}
        else:
            choices, target = mmlu_choices(row, position)
            prompt = (row["question"] + "\n" +
                      "\n".join(f"{LET[n]}. {choice}" for n, choice in enumerate(choices)) +
                      "\nRespond with exactly one character: the letter A, B, C or D. No explanation.")
            text = chat(args.endpoint, args.model, prompt, num_predict)
            pick = parse_mc(text, choices)
            detail = {"i": index, "subj": row["subject"], "pass": pick == target,
                      "pick": pick, "gold": target, "unparsed": pick < 0, "text": text}
        details.append(detail)
        atomic_json(checkpoint, {"identity": identity, "done": len(details), "details": details,
                                 "elapsed_s": previous_elapsed + time.monotonic() - started})
        print(f"{task} {len(details)}/{len(items)} idx={index}: {detail['pass']}", flush=True)
    return details, identity, previous_elapsed + time.monotonic() - started


def add_ollama_args(parser):
    parser.add_argument("--notes-dir", default="D:/qwen/notes")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")


def gsm_main(mode, runner, argv=None):
    parser = argparse.ArgumentParser(description=mode + " GSM8K evaluation")
    parser.add_argument("--model", required=mode != "harness", default="qwen3-f16-probe")
    parser.add_argument("--tag", required=mode != "harness", default="F16PROBE", type=safe_tag)
    parser.add_argument("--n", type=positive_int, default=50 if mode == "harness" else 200)
    parser.add_argument("--offset", type=nonnegative_int, default=0)
    if mode == "sweep":
        parser.add_argument("--precision", required=True)
    if mode == "gsm200":
        parser.add_argument("--npredict", type=positive_int, default=256)
    add_ollama_args(parser)
    args = parser.parse_args(argv)
    import datasets
    items = gsm_items(datasets.load_dataset("openai/gsm8k", "main", split="test"), args.offset, args.n)
    paths = {"sweep": (f"R2CKPT_GSMSW_{args.tag}_{args.offset}.json", f"RUN_GSMSWEEP_{args.tag}_{args.offset}.json"),
             "gsm200": (f"CKPT_GSM200_{args.tag}.json", f"RUN_QWEN3_{args.tag}.json"),
             "harness": (f"R2CKPT_{args.tag}_{args.offset}.json", f"RUN_HARNESS_{args.tag}_{args.offset}.json")}
    checkpoint, output = [Path(args.notes_dir) / name for name in paths[mode]]
    count = getattr(args, "npredict", 256)
    precision = getattr(args, "precision", "F16" if mode == "harness" else "unspecified")
    details, identity, elapsed = evaluate(args, items, "gsm", checkpoint, runner, count, precision=precision)
    score = sum(row["pass"] for row in details)
    result = {"model": args.model, "tag": args.tag, "precision": precision, "n": len(details),
              "gsm": score, "acc": round(score / len(details), 4), "offset": args.offset,
              "protocol": f"ollama-greedy-CoT-####-num_predict{count}", "run_identity": identity,
              "details": details, "elapsed_s": round(elapsed)}
    if mode == "gsm200":
        result["gsm200"] = score
    if mode == "harness":
        result["score"] = score
    atomic_json(output, result)
    print(f"GSM-{len(details)}: {score}/{len(details)} saved {output}")
