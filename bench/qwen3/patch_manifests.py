"""Patch manifest Qwen3 cho conform schema, KHONG bia dat.

Nguyen tac: cai gi khong chung minh duoc thi ghi null kem ly do.
Khong tauch hash, khong tauch timestamp, khong gan scorer khi khong biet.

Sinh ra D:/qwen_release/manifests_qwen3/ voi ban sao conform.
File goc trong notes/ khong bi dong vao.
"""
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

NOTES = Path("D:/qwen/notes")
MODELS = Path("D:/qwen/models")
OUT = Path("D:/qwen_release/manifests_qwen3")

# Ban do: ten manifest -> script da sinh ra nhan 'pass'.
# Chi nhung mapping ma ta that su chung minh duoc bang cach doi chieu
# truc 'protocol' + 'details' cua tung file.
SCORERS = {
    "mmlu_ollama": ("bench/qwen3/mmlu_any.py", "D:/qwen/bench/mmlu_any.py"),
    "gsm_ollama": ("bench/qwen3/gsm_sweep.py", "D:/qwen/bench/gsm_sweep.py"),
    "gsm_harness": ("bench/qwen3/harness_probe.py", "D:/qwen/bench/harness_probe.py"),
}

LIM_GPU = ("Diem khong tai lap duoc: can GGUF 2.5GB + Ollama + ~20-140 phut GPU. "
           "CI chi verify tinh toan, khong tao lai diem.")
LIM_SCRIPT = ("Script goc da bien mat khoi dia (script_sha256=689631b78649 khong "
              "khop file nao). Manifest giu nguyen ket qua cu, khong kha nang "
              "sinh lai.")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_cached(p: Path, cache):
    k = str(p)
    if k not in cache:
        cache[k] = sha(p)
    return cache[k]


def detail_schema(det):
    """Doi chieu key cua details[0] de biet ho so duoc gi.

    Ba ho khac nhau:
      mmlu    : i, subj, gold          -- co dap an
      gsm     : i, exp, got            -- co dap an
      typed   : id, type, raw, pass    -- rescore, khong co i
      harness : i, pass                -- CHI co chi so, KHONG luu dap an
    """
    k = set(det[0].keys())
    if "subj" in k and "gold" in k:
        return "mmlu"
    if "exp" in k and "got" in k:
        return "gsm"
    if "raw" in k and "id" in k:
        return "typed"
    if set(k) == {"i", "pass"}:
        return "harness"
    return "unknown:" + ",".join(sorted(k))


SCOPE = {
    "mmlu": "sha256 cua (i, mon, gold) da cham thuc su.",
    "gsm": "sha256 cua (i, dap-an-mong-doi) da cham thuc su.",
    "typed": "sha256 cua (id, type, raw) -- rescore tren the he co dong, "
             "khong co chi so i.",
    "harness": "CHI SHA256 CUA (i) -- manifest khong luu dap an mong doi, "
               "nen hash chi pin duoc TAP CAU, KHONG pin duoc noi dung cau. "
               "Khong du de chung minh dataset khong doi.",
    "unknown": "khong xac dinh duoc schema.",
}


def content_hash(details, kind):
    """Hash noi dung da cham thuc su.

    KHONG gom nhan 'pass' vao: doi so lieu cham se doi hash, va do khong
    con la hash dataset nua.
    """
    h = hashlib.sha256()
    if kind == "typed":
        for r in sorted(details, key=lambda r: str(r["id"])):
            h.update(("%s|%s|%s\n" % (r["id"], r.get("type", ""),
                                      r.get("raw", ""))).encode("utf-8"))
        return h.hexdigest()
    if kind == "harness":
        for r in sorted(details, key=lambda r: r["i"]):
            h.update(("i=%d\n" % r["i"]).encode("utf-8"))
        return h.hexdigest()
    for r in sorted(details, key=lambda r: r["i"]):
        if kind == "mmlu":
            h.update(("%d|%s|%s\n" % (r["i"], r.get("subj", ""),
                                      r.get("gold", ""))).encode("utf-8"))
        else:
            h.update(("%d|%s\n" % (r["i"], r.get("exp", ""))).encode("utf-8"))
    return h.hexdigest()


def pick_scorer(name, d, kind):
    """Chon script sinh nhan 'pass', chi khi bang chung du manh."""
    if kind == "mmlu":
        return "mmlu_ollama"
    if kind == "gsm":
        return "gsm_harness" if name.startswith("RUN_HARNESS_") else "gsm_ollama"
    if kind == "harness":
        return "gsm_harness"
    return None  # typed: script rescore V3.3, kiem chuc chua xac nhan


OLLAMA = Path("C:/Users/ACER/AppData/Local/Programs/Ollama/ollama.exe")


def ollama_blob_sha(model):
    """Ten model Ollama -> sha256 cua GGUF.

    Cach nay khong doan: Modelfile tro toi blob, ten blob CHINH la sha256
    cua file GGUF. Chuoi bang chung: model -> blob -> sha256 -> file dia.
    """
    import subprocess
    if not model or not OLLAMA.exists():
        return None
    try:
        out = subprocess.run([str(OLLAMA), "show", "--modelfile", model],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if line.strip().upper().startswith("FROM") and "sha256-" in line:
            return line.strip().split("sha256-")[-1].strip()
    return None


def build_disk_index(cache):
    """sha256 -> file, cho cac .gguf con tren dia."""
    idx = {}
    for f in sorted(MODELS.glob("*.gguf")):
        idx.setdefault(sha_cached(f, cache), f)
    return idx


def resolve_gguf(d, disk_idx):
    """Tra ve (ten_file, path, sha256, nguon) neu xac dinh duoc artifact."""
    pr = d.get("provenance") or {}
    # 1. manifest da tro san ten file
    for c in (pr.get("gguf"), d.get("gguf")):
        if not c:
            continue
        f = Path(c)
        if not f.is_absolute():
            f = MODELS / f.name
        if f.exists() and f.suffix == ".gguf":
            h = f.sha if hasattr(f, "sha") else None
            return f.name, f, None, "manifest.provenance.gguf"
    # 2. truy nguoc qua blob Ollama
    h = ollama_blob_sha(d.get("model"))
    if h and h in disk_idx:
        f = disk_idx[h]
        return f.name, f, h, "ollama-modelfile-FROM-blob"
    return None, None, h, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    disk_idx = build_disk_index(cache)
    report = {"conform": 0, "total": 0, "rows": []}

    for p in sorted(NOTES.glob("RUN_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("details"):
            continue
        report["total"] += 1
        det = d["details"]
        kind = detail_schema(det)
        ch = content_hash(det, kind)
        sc = pick_scorer(p.name, d, kind)

        gname, gpath, gsha, gsrc = resolve_gguf(d, disk_idx)
        pr = dict(d.get("provenance") or {})
        if pr.get("llama_cpp"):
            # giu ten goi, bo duong dan build tren may lab
            pr["llama_cpp"] = Path(str(pr["llama_cpp"])).name
        if gpath is not None:
            h = sha_cached(gpath, cache)
            pr["gguf"] = gname
            pr["gguf_sha256"] = h
            pr["gguf_bytes"] = gpath.stat().st_size
            pr["gguf_resolved_via"] = gsrc
            # doi chieu voi hash ghi san trong manifest goc
            old = (d.get("provenance") or {}).get("gguf_sha256")
            if old:
                pr["gguf_sha256_matches_original"] = (old == h)
        elif gsha:
            pr.setdefault("gguf_available", False)
            pr["gguf_sha256_from_ollama_blob"] = gsha
            pr["gguf_note"] = ("Blob Ollama tro toi file GGUF khong con trong "
                               "models/; co sha256 nhung khong kiem chung duoc noi "
                               "dung file.")
        else:
            pr.setdefault("gguf_available", False)
            pr["gguf_note"] = ("Khong xac dinh duoc artifact: manifest khong tro "
                               "ten file, va ten model Ollama khong con.")

        rel, shav = SCORERS.get(sc, (None, None))
        scorer_block = {
            "script": rel,
            "sha256": None,
            "status": None,
        }
        if sc and Path(shav).exists():
            scorer_block["sha256"] = sha_cached(Path(shav), cache)
            scorer_block["status"] = "verified-on-disk"
        elif sc:
            scorer_block["status"] = "script-khong-ton-tai"
        else:
            scorer_block["status"] = "khong-xac-dinh-duoc"

        mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
        # Khong dua duong dan may-lab len repo public. Giu basename, va
        # ghi ro day la thu muc dia (HF) chu khong phai ten trong Ollama.
        raw_model = d.get("model") or (d.get("provenance") or {}).get("gguf")
        if not raw_model:
            model, mkind = None, None
        elif re.match(r"^[A-Za-z]:[\\/]", raw_model):
            model, mkind = Path(raw_model).name, "local-directory"
        elif "/" in raw_model or "\\" in raw_model:
            model, mkind = Path(raw_model).name, "path"
        else:
            model, mkind = raw_model, "registry-name"
        out = {
            "run_id": hashlib.sha256(
                (p.name + ch).encode("utf-8")).hexdigest()[:16],
            "model": model,
            "model_path_kind": mkind,
            "protocol": d.get("protocol", "khong-ghi"),
            "dataset_sha256": ch,
            "detail_schema": kind,
            "dataset_sha256_scope": SCOPE.get(kind, SCOPE["unknown"]),
            "scorer": scorer_block["script"],
            "scorer_sha256": scorer_block["sha256"],
            "scorer_status": scorer_block["status"],
            "timestamp": mtime.isoformat().replace("+00:00", "Z"),
            "timestamp_source": "file-mtime-cua-manifest, KHONG phai gio chay thoi",
            "provenance": pr,
            "regenerable": False,
            "limitation": LIM_GPU,
            "scores": {k: v for k, v in d.items()
                       if isinstance(v, (int, float, bool, str))
                       and k not in ("protocol", "model", "gguf", "precision",
                                     "tag", "note")},
            "details": det,
        }
        if "precision" in d:
            out["scores"]["precision"] = d["precision"]
        if "note" in d:
            out["note"] = d["note"]

        (OUT / p.name).write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        report["conform"] += 1
        report["rows"].append({
            "name": p.name,
            "run_id": out["run_id"],
            "scorer_status": out["scorer_status"],
            "gguf": pr.get("gguf") or None,
            "gguf_verified": bool(gpath),
            "gguf_via": gsrc,
            "gguf_matches_original": pr.get("gguf_sha256_matches_original"),
            "dataset_sha256": ch[:12],
        })

    (OUT / "_PATCH_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Da patch %d manifest -> %s" % (report["conform"], OUT))
    rows = report["rows"]
    bad = [r for r in rows if not r["gguf_verified"]]
    ns = [r for r in rows if r["scorer_status"] != "verified-on-disk"]
    mm = [r for r in rows if r["gguf_matches_original"] is False]
    print("  artifact xac dinh + hash: %d / %d" % (len(rows) - len(bad), len(rows)))
    for k, lab in ((None, ""), ("manifest.provenance.gguf", "tu manifest"),
                   ("ollama-modelfile-FROM-blob", "qua blob Ollama")):
        n = sum(1 for r in rows if r["gguf_via"] == k)
        print("     %-32s %d" % (lab or "khong xac dinh", n))
    print("  hash khop voi ban goc: %d/%d (lech: %d)" % (
        len(rows) - len(mm), sum(1 for r in rows if r["gguf_matches_original"] is not None),
        len(mm)))
    print("  scorer xac minh tren dia: %d / %d" % (len(rows) - len(ns), len(rows)))
    for r in ns:
        print("     scorer: %-36s %s" % (r["name"], r["scorer_status"]))
    for r in bad:
        print("     artifact: %-34s %s" % (r["name"], "khong xac dinh duoc"))


if __name__ == "__main__":
    main()
