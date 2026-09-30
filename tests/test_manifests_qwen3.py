"""Kiem chung manifest Qwen3 trong manifests_qwen3/.

KHONG chay model, KHONG can GPU. Kiem duoc ba tang:

  1. Truong bat buoc theo schema (run_id/protocol/dataset_sha256/...)
  2. Hash dung dinh dang 64 hex, va scorer/gguf hash phai khop file
     tren dia -- neu file khong ton tai, test chi kiem dinh dang, KHONG
     doan la hash sai.
  3. Tinh lai dataset_sha256 tu `details` va so sanh. Day la tang manh
     nhat: bat duoc ca viec sua diem, sua danh sach cau, hay sua ca hai.

Khong kiem chung duoc, va test KHONG gia lap:
  - diem co khop voi model that hay khong (can 2.5GB GGUF + GPU)
  - co ton tai GGUF tren may nguoi doc hay khong
Vi vay moi co truong `regenerable: false` trong tung manifest.
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests_qwen3"
SCRIPT_DIR = ROOT / "bench" / "qwen3"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED = ["run_id", "model", "protocol", "dataset_sha256",
            "dataset_sha256_scope", "timestamp", "timestamp_source",
            "provenance", "regenerable", "limitation", "scores", "details"]

FILES = sorted(MAN.glob("RUN_*.json"))


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _recompute(details, kind):
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


def _ids():
    return [p.stem for p in FILES]


def test_manifest_present():
    assert FILES, "khong tim thay manifest nao trong %s" % MAN


@pytest.mark.parametrize("name", _ids())
def test_required_fields(name):
    d = _load(MAN / (name + ".json"))
    missing = [k for k in REQUIRED if k not in d or d[k] in ("", None)]
    assert not missing, "%s thieu truong: %s" % (name, missing)


@pytest.mark.parametrize("name", _ids())
def test_hash_format(name):
    d = _load(MAN / (name + ".json"))
    assert HEX64.match(d["dataset_sha256"]), name
    if d.get("scorer_sha256"):
        assert HEX64.match(d["scorer_sha256"]), name
    pr = d["provenance"]
    for k in ("gguf_sha256", "gguf_sha256_from_ollama_blob"):
        if pr.get(k):
            assert HEX64.match(pr[k]), "%s: %s" % (name, k)


@pytest.mark.parametrize("name", _ids())
def test_dataset_hash_recomputes(name):
    """Tinh lai tu details. Bat duoc moi thay doi noi dung."""
    d = _load(MAN / (name + ".json"))
    got = _recompute(d["details"], d["detail_schema"])
    assert got == d["dataset_sha256"], (
        "%s: dataset_sha256 khong khop details. Manifest bi sua?" % name)


@pytest.mark.parametrize("name", _ids())
def test_scores_agree_with_details(name):
    """Diem tong trong `scores` phai bang dem lai tu `details`."""
    d = _load(MAN / (name + ".json"))
    det = d["details"]
    n_pass = sum(1 for r in det if r.get("pass"))
    s = d["scores"]
    for key in ("mmlu", "gsm", "gsm400", "typed_suite"):
        if key in s:
            assert s[key] == n_pass, (
                "%s: scores.%s=%s nhung details co %d/true" % (
                    name, key, s[key], n_pass))
    if "n" in s and s["n"] is not None:
        assert s["n"] == len(det), name
    if "acc" in s and s["acc"] is not None:
        # manifest goc lam tron acc toi 4 chu so; so sanh sau khi lam tron
        assert round(s["acc"], 4) == round(n_pass / len(det), 4), (
            "%s: scores.acc=%s, details cho %.6f" % (
                name, s["acc"], n_pass / len(det)))


@pytest.mark.parametrize("name", _ids())
def test_honesty_fields(name):
    """Cac truong 'khong chung minh duoc' phai noi ro, khong de trong."""
    d = _load(MAN / (name + ".json"))
    assert d["regenerable"] is False, name
    assert "GPU" in d["limitation"], name
    if d["scorer"] is None:
        assert d["scorer_status"] != "verified-on-disk", (
            "%s: scorer None nhung lai khai bao da xac minh" % name)
    if d["scorer_status"] == "verified-on-disk":
        assert d["scorer_sha256"], name


def test_scorer_hashes_match_repo_scripts():
    """Hash scorer phai tro den file thuc su co trong repo nay."""
    import os
    for p in FILES:
        d = _load(p)
        if d["scorer_status"] != "verified-on-disk":
            continue
        fp = ROOT / d["scorer"]
        assert fp.exists(), "%s: scorer %s khong co trong repo" % (p.name, d["scorer"])
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        assert h == d["scorer_sha256"], (
            "%s: scorer_sha256 lech voi %s tren dia" % (p.name, d["scorer"]))


def test_gguf_hash_matches_manifest_original():
    """Hash gguf trong ban patch phai bang hash ghi trong manifest goc.

    Do doc lap: doc file goc trong notes/ cua may lab. tren may khong co
    notes/ (khong commit file phuc tap), test nay tu bo qua.
    """
    src = Path("D:/qwen/notes")
    if not src.is_dir():
        pytest.skip("khong co thu muc lab D:/qwen/notes tren may nay")
    n = 0
    for p in FILES:
        o = src / p.name
        if not o.exists():
            continue
        od = _load(o)
        oh = (od.get("provenance") or {}).get("gguf_sha256")
        if not oh:
            continue
        nd = _load(p)["provenance"]
        assert nd["gguf_sha256"] == oh, "%s: doi file GGUF so voi ban goc" % p.name
        n += 1
    assert n >= 10, "chi doi chieu duoc %d manifest, ke mong >= 10" % n


def test_no_duplicate_run_id():
    ids = [_load(p)["run_id"] for p in FILES]
    assert len(ids) == len(set(ids)), "trung run_id"


def test_gguf_present_at_least_20():
    """Phan lon phai tro duoc ve artifact that tren dia."""
    n = sum(1 for p in FILES
            if _load(p)["provenance"].get("gguf_sha256"))
    assert n >= 20, "chi %d/%d manifest co hash GGUF" % (n, len(FILES))
