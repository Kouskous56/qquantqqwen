"""Patch manifest Qwen3 cho conform schema, KHONG bia dat.

Nguyen tac: cai gi khong chung minh duoc thi ghi null kem ly do.
Khong tauch hash, khong tauch timestamp, khong gan scorer khi khong biet.

Sinh ra D:/qwen_release/manifests_qwen3/ voi ban sao conform.
File goc trong notes/ khong bi dong vao.
"""
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .common import atomic_json
else:
    from common import atomic_json

ROOT = Path(__file__).resolve().parents[2]
FALLBACK_FILE = ROOT / "manifests_qwen3" / "_MODEL_FILE_FALLBACK.json"

NOTES = Path("D:/qwen/notes")
MODELS = Path("D:/qwen/models")
OUT = Path("D:/qwen_release/manifests_qwen3")

# Ban do: ten logic -> (duong dan trong repo, duong dan lab)
SCORERS = {
    "mmlu_ollama": ("bench/qwen3/mmlu_any.py", "D:/qwen/bench/mmlu_any.py"),
    "gsm_ollama": ("bench/qwen3/gsm_sweep.py", "D:/qwen/bench/gsm_sweep.py"),
    "gsm_harness": ("bench/qwen3/harness_probe.py", "D:/qwen/bench/harness_probe.py"),
    "eval_r2": ("bench/qwen3/eval_round2.py", "D:/qwen/bench/eval_round2.py"),
    "gsm200_r1": ("bench/qwen3/gsm200_ollama.py", "D:/qwen/bench/gsm200_ollama.py"),
    "gsm_fp16": ("bench/qwen3/eval_qwen3_gsm.py", "D:/qwen/bench/eval_qwen3_gsm.py"),
}

# Cac ban gsm_sweep.py da biet, phan biet bang DAC DIEM KIEM CHUNG DUOC
# (khong phai bang tri nho): dong OUT va truong trong det.append.
#   4576 = git 6d86908: OUT khong offset, details {i,pass}       -> Q6
#   7e5e = repo hien tai: OUT co offset, {i,exp,got}, KHONG text -> Q5
#   b1cb = lab hien tai: OUT co offset, {i,exp,got,text}         -> Q3
GSM_SWEEP_VERSIONS = {
    "v_orig": {
        "hash": "457690e0a96bfe27ec2c9a9ee4602d28c6f345af7971021ae64d37089f91f107",
        "features": "OUT khong offset; details {i,pass}",
        "source": "git cat-file 6d86908:bench/qwen3/gsm_sweep.py",
    },
    "v_exp": {
        "hash": "7e5ec003777b9a2ffb598b49bb0d3f0eb35b4575e17b5dacf72ed9480ca1953a",
        "features": "OUT co offset; details {i,exp,got}, khong text",
        "source": "ban repo truoc khi copy lab (da xac minh dac diem truc tiep)",
    },
    "v_text": {
        "hash": "b1cb7dc35a316fd0cfe06105d8e56f61ebf4956620d194340859d0023e2ca1ae",
        "features": "OUT co offset; details {i,exp,got,text}",
        "source": "lab hien tai (da xac minh dac diem truc tiep)",
    },
}

LIM_GPU = ("Diem khong tai lap duoc: can GGUF 2.5GB + Ollama + ~20-140 phut GPU. "
           "CI chi verify tinh toan, khong tao lai diem.")
LIM_SCRIPT = ("Script goc da bien mat khoi dia (script_sha256=689631b78649 khong "
              "khop file nao). Manifest giu nguyen ket qua cu, khong kha nang "
              "sinh lai.")

# Script da bi sua SAU khi sinh ra so lieu. Manifest phai giu hash luc chay
# (nguon cua so lieu), khong phai hash hien tai -- neu doi hash thi se noi
# sai rang ban da sua da sinh ra cac diem nay.
# Hash cua ban script LUC CHAY cho cac script chi co mot ban da biet.
# Lay tu git HEAD truoc khi sua portability (doi chieu truc tiep, khong nho).
# gsm_sweep.py khong nam day vi no co 3 ban (GSM_SWEEP_VERSIONS).
RUN_TIME_HASHES = {
    "bench/qwen3/mmlu_any.py":
        "2db326463998922c6a29a692d60805fd10640b987e0ab0b84c8aba3051c1e833",
    "bench/qwen3/harness_probe.py":
        "9cbd74fc9b1beaec5f9e97299ef1783562be524d7336662654a3241ee9c6fdb9",
    "bench/qwen3/eval_round2.py":
        "d8998bae091cf308fbe6494c1b3dfe7465d380eabf33adbd96b11dea9e258f7f",
    "bench/qwen3/gsm200_ollama.py":
        "a7cce0be3e93425035477473746301d34d562c3dd55a57d2a1e75c091f95e1dc",
    "bench/qwen3/eval_qwen3_gsm.py":
        "23533bb89b4f3d6b9c6df888eb99c099f35b6d24fb68de708281812ab874f817",
}

PORTABILITY_NOTE = (
    "Lan sua portability (them --notes-dir/--endpoint, mac dinh giu hanh vi "
    "cu): chi them tham so dong lenh, khong doi logic cham, prompt, trich "
    "so. Du lieu cu sinh bang ban cu nen scorer_sha256 giu nguyen.")

POST_RUN_FIXES = {
    "bench/qwen3/gsm_sweep.py": {
        # Lay tu git cat-file 6d86908 -- ban da sinh ra so lieu. Khong do
        # tu file hien tai, vI file da bi sua.
        "hash_at_run": "457690e0a96bfe27ec2c9a9ee4602d28c6f345af7971021ae64d37089f91f107",
        "reason": ("Ten file manifest khong kem offset (OUT khong co _{a.offset}, "
                   "con CK thi co), nen block offset=200 de file manifest cua "
                   "block offset=0. DU LIEU KHONG MAT: checkpoint giu 200 cau moi "
                   "block, va merge_gsm_blocks.py gop lai duoc, kiem tra trung i. "
                   "Chi 1 manifest bi de; duoc ghi lai tu checkpoint."),
        "affects_data": False,
    },
}


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
    if not det or not all(isinstance(row, dict) for row in det):
        raise ValueError("details must be a nonempty list of objects")
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
    if kind in ("gsm", "harness"):
        # harness_probe.py xuat ban RUN_HARNESS_*; gsm_sweep.py xuat ban con lai
        return "gsm_harness" if name.startswith("RUN_HARNESS_") else "gsm_ollama"
    return None  # typed: script rescore V3.3, kiem chuc chua xac nhan


def attribute(name, d, kind, det0):
    """Tra ve (sc_key, version_key, evidence). Khong doan.

    Moi mapping phai co it nhat 2 bang chung doc lap:
      E1: schema cua details khop voi cau append trong script
      E2: tag khop voi quy uoc dat ten / mau OUT cua script
      E3: mtime script cu hon manifest (yeu, chi ho tro)
    Neu khong du 2 bang chung: tra (None, None, ly-do).
    """
    tag = d.get("tag") or ""
    keys = set(det0.keys())

    # --- R2: tag R2_* + schema khop eval_round2.py ---
    if tag.startswith("R2_"):
        if kind in ("gsm",):
            return ("eval_r2", None,
                    "E1: details {i,exp,got} khop eval_round2.py L155-156; "
                    "E2: tag R2_* + checkpoint R2CKPT_GSM_* do script nay ghi")
        if kind == "mmlu":
            return ("eval_r2", None,
                    "E1: details khop mdet.append cua eval_round2.py; "
                    "E2: tag R2_* + checkpoint R2CKPT_MMLU_* do script nay ghi")
        return (None, None, "tag R2_ nhung schema la khong biet")

    # --- sweep GSM Q2/Q3/Q4/Q5/Q6: tag *KMi/*Ki + offset, phan biet ban ---
    # CAN kind vi tag Q5KMi/Q6Ki dung cho ca GSM lan MMLU.
    if kind in ("gsm", "harness") and (
            tag in ("Q2Ki", "Q3KMi", "Q40", "Q4KS", "Q5KMi", "Q6Ki")
            or name.startswith("RUN_GSMSWEEP_Q")):
        if "text" in keys:
            return ("gsm_ollama", "v_text",
                    "E1: details co 'text', chi ban v_text ghi truong nay; "
                    "E2: tag %s + offset, khop mau OUT co-offset" % tag)
        if "exp" in keys and "got" in keys:
            return ("gsm_ollama", "v_exp",
                    "E1: details {i,exp,got} khong text, khop ban v_exp; "
                    "E2: tag %s + offset" % tag)
        if keys == {"i", "pass"}:
            return ("gsm_ollama", "v_orig",
                    "E1: details {i,pass}, khop ban goc v_orig; "
                    "E2: tag %s" % tag)
        return (None, None, "tag sweep nhung schema la")

    # --- sweep MMLU F16/Q5/Q6 (khong phai R2) ---
    if kind == "mmlu" and tag in ("F16", "Q3KMi", "Q5KMi", "Q6Ki"):
        return ("mmlu_ollama", None,
                "E1: details khop det.append cua mmlu_any.py; "
                "E2: tag %s, khong phai R2_" % tag)

    # --- R1 GSM200: schema co 'chars' ---
    if "chars" in keys:
        return ("gsm200_r1", None,
                "E1: details co 'chars', chi gsm200_ollama.py L67 ghi; "
                "E2: mau OUT RUN_QWEN3_{tag} khop ten file")

    # --- FP16 baseline: truong gsm50_256 ---
    if "gsm50_256" in d:
        return ("gsm_fp16", None,
                "E1: truong gsm50_256, chi eval_qwen3_gsm.py L52 ghi; "
                "E2: model la thu muc FP16, khop dau vao transformers")

    # --- harness probe ---
    if name.startswith("RUN_HARNESS_") and kind in ("harness", "gsm"):
        return ("gsm_harness", None,
                "E1: details {i,pass} khop harness_probe.py; "
                "E2: ten file RUN_HARNESS_*")

    return (None, None, "khong du 2 bang chung doc lap")


def post_run_fix_for(tag, sc):
    """Ban script da sua sau khi chay chi ap dung cho manifest do script do sinh."""
    if sc != "gsm_ollama":
        return None
    return POST_RUN_FIXES.get("bench/qwen3/gsm_sweep.py")


OLLAMA = Path("C:/Users/ACER/AppData/Local/Programs/Ollama/ollama.exe")


def ollama_blob_sha(model):
    """Ten model Ollama -> sha256 cua GGUF.

    Cach nay khong doan: Modelfile tro toi blob, ten blob CHINH la sha256
    cua file GGUF. Chuoi bang chung: model -> blob -> sha256 -> file dia.
    """
    import subprocess
    if not model or not OLLAMA:
        return None
    try:
        result = subprocess.run([str(OLLAMA), "show", "--modelfile", model],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        match = re.fullmatch(r'\s*FROM\s+.*sha256-([0-9a-fA-F]{64})"?\s*', line, re.I)
        if match:
            return match.group(1).lower()
    return None


def build_disk_index(cache):
    """sha256 -> file, cho cac .gguf con tren dia."""
    idx = {}
    for f in sorted(MODELS.glob("*.gguf")):
        idx.setdefault(sha_cached(f, cache), f)
    return idx


# Anh xa ten model Ollama doi R1 -> file GGUF, cho truong hop model khong
# con trong Ollama nhung file con tren dia. Moi anh xa da duoc xac minh
# bang `ollama show --modelfile` (blob sha256 trung hash file) truoc khi
# model bi xoa. Hash van duoc tinh lai tu file that tren dia moi lan chay,
# nen bang chung khong yeu di -- chi duong di doi tu blob sang ten file.
def model_file_fallback():
    """Read the published single source of truth, rejecting path traversal."""
    data = json.loads(FALLBACK_FILE.read_text(encoding="utf-8"))
    mapping = {key: value for key, value in data.items() if not key.startswith("_")}
    for key, value in mapping.items():
        if (not isinstance(value, str) or not value.lower().endswith(".gguf")
                or "/" in value or "\\" in value or ":" in value):
            raise ValueError(f"Invalid GGUF fallback for {key}: {value!r}")
    return mapping


def resolve_gguf(d, disk_idx):
    """Tra ve (ten_file, path, sha256, nguon) neu xac dinh duoc artifact."""
    pr = d.get("provenance") or {}
    # 1. manifest da tro san ten file
    for c in (pr.get("gguf"), d.get("gguf")):
        if not c:
            continue
        # A Windows absolute path is not absolute to pathlib on Linux.
        normalized = str(c).replace("\\", "/")
        f = Path(normalized)
        if re.match(r"^[A-Za-z]:/", normalized) and not f.is_absolute():
            f = MODELS / normalized.rsplit("/", 1)[-1]
        if not f.is_absolute():
            f = MODELS / f.name
        if f.is_file() and f.suffix.lower() == ".gguf":
            return f.name, f, None, "manifest.provenance.gguf"
    # 2. truy nguoc qua blob Ollama
    h = ollama_blob_sha(d.get("model"))
    if h and h in disk_idx:
        f = disk_idx[h]
        return f.name, f, h, "ollama-modelfile-FROM-blob"
    # 3. anh xa legacy cho model R1 da xoa (file con, hash tinh lai)
    legacy = model_file_fallback().get(d.get("model") or "")
    if legacy:
        f = MODELS / legacy
        if f.is_file():
            return f.name, f, None, "legacy-model-to-file-map"
    return None, None, h, None


def recorded_scorer(data):
    """Use explicit new-run source identity instead of historical heuristics."""
    identities = data.get("run_identities") or [data.get("run_identity")]
    if not isinstance(identities, list):
        raise ValueError("run_identities must be a list")
    sources = [item.get("sources") for item in identities if isinstance(item, dict)]
    if not sources:
        return None
    if any(item != sources[0] for item in sources):
        raise ValueError("Run blocks were generated by different sources")
    sources = sources[0]
    if not isinstance(sources, dict) or not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in sources.values()):
        raise ValueError("Invalid recorded source digest")
    runners = {Path(rel).name: rel for rel, _ in SCORERS.values()}
    names = set(sources) & set(runners)
    if len(names) != 1 or set(sources) != names | {"common.py"}:
        raise ValueError("Unknown runner or missing shared source in run identity")
    name = names.pop()
    rel, digest = runners[name], sources[name]
    verified = all(sha(ROOT / "bench" / "qwen3" / filename) == value
                   for filename, value in sources.items())
    return {"script": rel, "sha256": digest,
            "status": "verified-on-disk" if verified else "source-digest-recorded",
            "generator_evidence": "Explicit source hashes recorded by the new-run identity",
            "dependencies": {"bench/qwen3/common.py": sources["common.py"]}}


def main(argv=None):
    global NOTES, MODELS, OUT, OLLAMA, FALLBACK_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes-dir", type=Path, default=NOTES)
    parser.add_argument("--models-dir", type=Path, default=MODELS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--fallback-file", type=Path, default=FALLBACK_FILE)
    parser.add_argument("--ollama", default=shutil.which("ollama") or str(OLLAMA))
    args = parser.parse_args(argv)
    NOTES, MODELS, OUT = args.notes_dir, args.models_dir, args.output_dir
    OLLAMA, FALLBACK_FILE = args.ollama, args.fallback_file
    if not NOTES.is_dir():
        parser.error(f"notes directory does not exist: {NOTES}")
    if NOTES.resolve() == OUT.resolve():
        parser.error("output directory must differ from the source notes directory")
    if OUT.resolve() == (ROOT / "manifests_qwen3").resolve():
        parser.error("published manifests are frozen; choose a separate --output-dir")
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
        sc, ver, ev = attribute(p.name, d, kind, det[0])

        gname, gpath, gsha, gsrc = resolve_gguf(d, disk_idx)
        pr = dict(d.get("provenance") or {})
        if pr.get("llama_cpp"):
            # giu ten goi, bo duong dan build tren may lab
            pr["llama_cpp"] = Path(str(pr["llama_cpp"])).name
        if gpath is not None:
            h = sha_cached(gpath, cache)
            old = (d.get("provenance") or {}).get("gguf_sha256")
            if old and old != h:
                raise ValueError(f"{p.name}: GGUF differs from recorded SHA-256; refusing to reattribute scores")
            pr["gguf"] = gname
            pr["gguf_sha256"] = h
            pr["gguf_bytes"] = gpath.stat().st_size
            pr["gguf_resolved_via"] = gsrc
            # doi chieu voi hash ghi san trong manifest goc
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
        # CHOT DON DIEU: khong bao gio ghi de hash GGUF tot bang ket qua kem.
        # Patcher xay tu manifest goc moi lan, nen khi moi truong suy giam
        # (vd Ollama serve chet, khong truy vet blob duoc) lan chay se mat
        # hash da xac minh. Truong hop do da xay ra that: 13 hash bien mat
        # trong mot lan chay lai. Tu nay: giu hash cu + ghi ro nguon.
        if not pr.get("gguf_sha256"):
            _prev = OUT / p.name
            if _prev.exists():
                try:
                    _ppr = json.loads(_prev.read_text(encoding="utf-8"))
                    _pprov = _ppr.get("provenance") or {}
                except (ValueError, OSError) as exc:
                    raise ValueError(f"Cannot preserve prior provenance from {_prev}") from exc
                if _pprov.get("gguf_sha256"):
                    pr["gguf"] = _pprov.get("gguf", pr.get("gguf"))
                    pr["gguf_sha256"] = _pprov["gguf_sha256"]
                    pr["gguf_bytes"] = _pprov.get("gguf_bytes",
                                                  pr.get("gguf_bytes"))
                    pr["gguf_resolved_via"] = _pprov.get(
                        "gguf_resolved_via", "carried-over-previous-run")
                    pr["gguf_carried_over"] = True
                    pr["gguf_carry_reason"] = (
                        "Lan chay nay khong phan giai duoc artifact "
                        "(vd serve tat); giu hash da xac minh tu ban truoc "
                        "thay vi de mat. Khong phai bang chung moi.")

        rel, shav = SCORERS.get(sc, (None, None))
        scorer_block = {
            "script": rel,
            "sha256": None,
            "status": None,
            "generator_evidence": ev,
        }
        if sc and shav and Path(shav).exists():
            cur = sha_cached(Path(shav), cache)
            if sc == "gsm_ollama" and ver:
                # moi manifest giu hash cua DUNG ban da sinh ra no,
                # xac dinh bang dac diem (khong phai tri nho)
                run_h = GSM_SWEEP_VERSIONS[ver]["hash"]
                scorer_block["sha256"] = run_h
                scorer_block["generator_version"] = ver
                scorer_block["generator_features"] = \
                    GSM_SWEEP_VERSIONS[ver]["features"]
                scorer_block["status"] = "verified-on-disk"
                if run_h != cur:
                    scorer_block["current_sha256"] = cur
                    scorer_block["modified_after_run"] = True
                    scorer_block["modification"] = (
                        "Script bi sua SAU khi sinh so lieu nay. "
                        "scorer_sha256 giu ban luc chay; current_sha256 la "
                        "ban hien tai. Chi tiet 3 lan sua: (1) ten file "
                        "manifest thieu offset nen block sau de block truoc; "
                        "(2) them truong exp/got/text vao details; "
                        "(3) portability: them --notes-dir/--endpoint, "
                        "mac dinh giu hanh vi cu, khong doi logic cham.")
            else:
                run_h = RUN_TIME_HASHES.get(rel)
                if run_h is None:
                    # khong biet ban luc chay: noi ro, khong doan
                    scorer_block["sha256"] = cur
                    scorer_block["status"] = "verified-on-disk"
                    scorer_block["version_note"] = (
                        "Lab khong co git; khong chung minh duoc file nay "
                        "giong ban luc chay. Bang chung la schema + tag + mtime "
                        "(xem generator_evidence). 'Khong biet sua' khac voi "
                        "'chung minh chua sua'.")
                elif run_h == cur:
                    scorer_block["sha256"] = cur
                    scorer_block["status"] = "verified-on-disk"
                else:
                    scorer_block["sha256"] = run_h
                    scorer_block["status"] = "verified-on-disk"
                    scorer_block["current_sha256"] = cur
                    scorer_block["modified_after_run"] = True
                    scorer_block["modification"] = PORTABILITY_NOTE
        elif sc:
            scorer_block["status"] = "script-khong-ton-tai"
        else:
            scorer_block["status"] = "khong-xac-dinh-duoc"
        explicit_scorer = recorded_scorer(d)
        if explicit_scorer:
            scorer_block = explicit_scorer

        mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
        # Khong dua duong dan may-lab len repo public. Giu basename, va
        # ghi ro day la thu muc dia (HF) chu khong phai ten trong Ollama.
        raw_model = d.get("model") or (d.get("provenance") or {}).get("gguf")
        if not raw_model:
            model, mkind = None, None
        elif re.match(r"^[A-Za-z]:[\\/]", raw_model):
            model, mkind = raw_model.replace("\\", "/").rsplit("/", 1)[-1], "local-directory"
        elif "/" in raw_model or "\\" in raw_model:
            model, mkind = raw_model.replace("\\", "/").rsplit("/", 1)[-1], "path"
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
            "current_sha256": scorer_block.get("current_sha256"),
            "modified_after_run": scorer_block.get("modified_after_run"),
            "modification": scorer_block.get("modification"),
            "generator_evidence": scorer_block.get("generator_evidence"),
            "generator_version": scorer_block.get("generator_version"),
            "generator_features": scorer_block.get("generator_features"),
            "version_note": scorer_block.get("version_note"),
            "scorer_dependencies": scorer_block.get("dependencies"),
            "timestamp": mtime.isoformat().replace("+00:00", "Z"),
            "timestamp_source": "file-mtime-cua-manifest, KHONG phai gio chay thoi",
            "provenance": pr,
            "dataset_source": d.get("dataset_source"),
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
        for key in ("run_identity", "run_identities"):
            if key in d:
                out[key] = d[key]

        atomic_json(OUT / p.name, out)
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

    atomic_json(OUT / "_PATCH_REPORT.json", report)
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
        sum(r["gguf_matches_original"] is True for r in rows),
        sum(1 for r in rows if r["gguf_matches_original"] is not None),
        len(mm)))
    print("  scorer xac minh tren dia: %d / %d" % (len(rows) - len(ns), len(rows)))
    for r in ns:
        print("     scorer: %-36s %s" % (r["name"], r["scorer_status"]))
    for r in bad:
        print("     artifact: %-34s %s" % (r["name"], "khong xac dinh duoc"))


if __name__ == "__main__":
    main()
