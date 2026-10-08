"""MMLU-200 tren F16 (lossless) -- la dong chieu bat buoc.

Ly do: tat ca so MMLU hien co (148/143/145/139/147) deu nam tren thang Q4.
Khong co so F16 nao thi khong noi duoc "Q4 khong dung MMLU", vi duoi day chi
la Q4 so voi Q4. Dong nay la mau chuan lossless de so sanh.

Dung cung 200 cau, cung 57 mon, cung phep hoan doi nhan A-D nhu
mmlu_ollama.py -> so sanh duoc truc tiep voi bang R2.
"""
import argparse
import json
import re
import time
import urllib.request

LET = "ABCD"
ARGS = None


def chat(content, n):
    body = json.dumps({"model": ARGS.model, "stream": False,
                       "messages": [{"role": "user", "content": content}],
                       "options": {"num_predict": n, "temperature": 0}}
                      ).encode()
    q = urllib.request.Request(EP + "/api/chat", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(q, timeout=900) as r:
        return json.load(r)["message"]["content"]


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gguf", default="")
    ap.add_argument("--precision", required=True,
                    help="F16 | Q4_K_M | ... de ghi vao manifest")
    ap.add_argument("--notes-dir", default="D:/qwen/notes",
                    help="thu muc ghi checkpoint + manifest (mac dinh giu duong lab cu)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:11434",
                    help="Ollama API root (mac dinh giu endpoint cu)")
    a = ARGS = ap.parse_args()

    EP = ARGS.endpoint.rstrip("/\\")
    NOTES = ARGS.notes_dir.rstrip("/\\")

    import datasets
    ck = f"{NOTES}/R2CKPT_MMLU_{a.tag}.json"
    out = f"{NOTES}/RUN_MMLU200_{a.tag}.json"
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

    try:
        det = json.load(open(ck, encoding="utf-8"))["details"]
        print(f"resume: {len(det)}", flush=True)
    except (FileNotFoundError, KeyError, ValueError):
        det = []

    t0 = time.time()
    for j, (i, r) in enumerate(items):
        if j < len(det):
            continue
        gold, target = r["answer"], j % 4
        others = [c for jj, c in enumerate(r["choices"]) if jj != gold]
        ch = others[:target] + [r["choices"][gold]] + others[target:]
        body = (r["question"] + "\n" +
                "\n".join(f"{LET[n]}. {c}" for n, c in enumerate(ch)) +
                "\nRespond with exactly one character: the letter A, B, C or D. "
                "No explanation.")
        pick = parse_mc(chat(body, 32), ch)
        det.append({"i": i, "subj": r["subject"], "pass": bool(pick == target),
                    "pick": pick, "gold": target, "unparsed": pick < 0})
        json.dump({"done": len(det), "details": det}, open(ck, "w"))
        if len(det) % 25 == 0:
            print(f"  {len(det)}/200 ({time.time()-t0:.0f}s)", flush=True)

    s = sum(d["pass"] for d in det)
    json.dump({"model": a.model, "tag": a.tag, "precision": a.precision,
               "gguf": a.gguf, "n": len(det), "mmlu": s,
               "acc": round(s / max(1, len(det)), 4),
               "n_subjects": len(subs),
               "unparsed": sum(1 for d in det if d.get("unparsed")),
               "protocol": "ollama-greedy-MC-one-letter-npredict32",
               "details": det, "elapsed_s": round(time.time() - t0)},
              open(out, "w"), indent=1)
    print(f"MMLU-200 [{a.precision}] {a.tag}: {s}/{len(det)} = "
          f"{100*s/len(det):.1f}%  ({time.time()-t0:.0f}s)")
    print(f"saved {out}")
