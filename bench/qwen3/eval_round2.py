"""Eval round 2: GSM-400 + MMLU-200 tren 5 artifact cung quy trinh.

Khac voi vong 1 o hai diem:
  1. Ghi sha256 cua file .gguf vao manifest -> chong lai provenance sai.
  2. Ghi ro llama.cpp version + co imatrix hay khong.
  3. Chay ca 2 block GSM (400 cau) + MMLU-200 trong mot lan, co checkpoint.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

LET = "ABCD"


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()


def chat(content, n):
    body = json.dumps({"model": ARGS.model, "stream": False,
                       "messages": [{"role": "user", "content": content}],
                       "options": {"num_predict": n, "temperature": 0}}
                      ).encode()
    q = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(q, timeout=900) as r:
        return json.load(r)["message"]["content"]


def extract(t):
    mm = re.search(r"\\boxed\{([^}]+)\}", t)
    if mm:
        return mm.group(1).replace(",", "").strip()
    mm = re.search(r"####\s*(-?[\d,.]+)", t)
    if mm:
        return mm.group(1).replace(",", "").strip()
    mg = re.findall(r"-?\d[\d,.]*", t)
    return mg[-1].replace(",", "") if mg else ""


def parse_mc(out, choices):
    t = out.strip()
    m = re.match(r"^[\*\s]*\(?([A-D])\)?[\)\.\:\s\*]", t)
    if m:
        return LET.index(m.group(1))
    m = re.search(r"answer[^A-D\n]{0,20}([A-D])\b", t, re.I)
    if m:
        return LET.index(m.group(1))
    for i, c in enumerate(choices):
        if c and len(c.strip()) > 2 and c.strip().lower() in t.lower()[:90]:
            return i
    m = re.search(r"\b([A-D])\b", t)
    if m:
        return LET.index(m.group(1))
    return -1


def load_ckpt(p):
    try:
        return json.load(open(p, encoding="utf-8"))["details"]
    except (FileNotFoundError, KeyError, ValueError):
        return []


def dump(p, done, det):
    json.dump({"done": done, "details": det}, open(p, "w"))


ARGS = None
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gguf", required=True, help="duong dan file .gguf de hash")
    ap.add_argument("--imatrix", default="unknown")
    ARGS = ap.parse_args()

    t0 = time.time()
    prov = {"gguf": os.path.basename(ARGS.gguf),
            "gguf_sha256": sha256(ARGS.gguf),
            "gguf_bytes": os.path.getsize(ARGS.gguf),
            "imatrix": ARGS.imatrix,
            "llama_cpp": "D:/qwen/tools/llama.cpp (build 2026-09-30)"}

    # ---------- MMLU-200 ----------
    import datasets
    mc = f"D:/qwen/notes/R2CKPT_MMLU_{ARGS.tag}.json"
    Qs = json.load(open("D:/qwen_release/data/v2/questions.json",
                        encoding="utf-8"))
    ds = datasets.load_dataset("cais/mmlu", "all", split="test")
    bysub = {}
    for i, r in enumerate(ds):
        bysub.setdefault(r["subject"], []).append((i, r))
    subs = sorted(bysub)
    items, k = [], 0
    while len(items) < 200:
        for s in subs:
            if len(items) >= 200:
                break
            if k < len(bysub[s]):
                items.append(bysub[s][k])
        k += 1
    mdet = load_ckpt(mc)
    for k, (i, r) in enumerate(items):
        if k < len(mdet):
            continue
        gold, target = r["answer"], k % 4
        others = [c for j, c in enumerate(r["choices"]) if j != gold]
        ch = others[:target] + [r["choices"][gold]] + others[target:]
        body = (r["question"] + "\n" +
                "\n".join(f"{LET[n]}. {c}" for n, c in enumerate(ch)) +
                "\nRespond with exactly one character: the letter A, B, C or D. "
                "No explanation.")
        pick = parse_mc(chat(body, 32), ch)
        mdet.append({"i": i, "subj": r["subject"], "pass": bool(pick == target),
                     "pick": pick, "gold": target, "unparsed": pick < 0})
        dump(mc, len(mdet), mdet)
    mscore = sum(d["pass"] for d in mdet)
    json.dump({"model": ARGS.model, "tag": f"R2_MMLU200_{ARGS.tag}",
               "provenance": prov, "n": len(mdet), "mmlu": mscore,
               "acc": round(mscore / max(1, len(mdet)), 4),
               "n_subjects": len(subs),
               "unparsed": sum(1 for d in mdet if d.get("unparsed")),
               "protocol": "ollama-greedy-MC-one-letter-npredict32",
               "details": mdet, "elapsed_s": round(time.time() - t0),
               "note": "GGUF Q4_K_M co imatrix; khong tron voi FP16 block"},
              open(f"D:/qwen/notes/RUN_QWEN3_R2_MMLU200_{ARGS.tag}.json", "w"),
              indent=1)
    print(f"R2-MMLU {ARGS.tag}: {mscore}/{len(mdet)}", flush=True)

    # ---------- GSM-400 (2 block, checkpoint rieng tung block) ----------
    gsm = datasets.load_dataset("openai/gsm8k", "main", split="test")
    gdet = []
    for off in (0, 200):
        gp = f"D:/qwen/notes/R2CKPT_GSM_{ARGS.tag}_{off}.json"
        blk = load_ckpt(gp)
        for j in range(200):
            if j < len(blk):
                continue
            i = off + j
            r = gsm[i]
            t = chat(r["question"] + "\nThink step by step, then end with: "
                     "#### <number>", 256)
            me = re.search(r"####\s*(-?[\d,.]+)", r["answer"])
            exp = me.group(1).replace(",", "") if me else ""
            blk.append({"i": i, "pass": extract(t) == exp,
                        "got": extract(t), "exp": exp})
            dump(gp, len(blk), blk)
        gdet += blk
    g = sum(d["pass"] for d in gdet)
    blkA = sum(d["pass"] for d in gdet if d["i"] < 200)
    blkB = sum(d["pass"] for d in gdet if d["i"] >= 200)
    json.dump({"model": ARGS.model, "tag": f"R2_GSM400_{ARGS.tag}",
               "provenance": prov, "n": len(gdet), "gsm": g, "gsm400": g,
               "blockA": blkA, "blockB": blkB,
               "protocol": "ollama-greedy-CoT-####-npredict256",
               "details": gdet, "elapsed_s": round(time.time() - t0),
               "note": "GGUF Q4_K_M co imatrix; khong tron voi FP16 block"},
              open(f"D:/qwen/notes/RUN_QWEN3_R2_GSM400_{ARGS.tag}.json", "w"),
              indent=1)
    print(f"R2-GSM400 {ARGS.tag}: A={blkA}/200 B={blkB}/200 "
          f"tong={g}/{len(gdet)}", flush=True)
    print(f"DONE {ARGS.tag}", flush=True)
