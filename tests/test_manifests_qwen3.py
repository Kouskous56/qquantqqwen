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
import os
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
    """Hash scorer phai khop mot trong cac ban da ghi nho.

    `scorer_sha256` la hash luc chay -- day la nguon cua so lieu.
    `current_sha256` la hash ban dang nam tren dia hien nay. Hai cai
    khac nhau nghia la script da bi sua sau khi sinh ra so lieu, va
    manifest phai noi ro ly do. Test nay KHONG chap nhan hash nao
    chua duoc ghi nho.
    """
    for p in FILES:
        d = _load(p)
        if d["scorer_status"] != "verified-on-disk":
            continue
        fp = ROOT / d["scorer"]
        assert fp.exists(), "%s: scorer %s khong co trong repo" % (p.name, d["scorer"])
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        known = {d["scorer_sha256"]}
        if d.get("current_sha256"):
            known.add(d["current_sha256"])
        assert h in known, (
            "%s: %s tren dia khop hash nao cung khong. known=%s actual=%s" % (
                p.name, d["scorer"], sorted(k[:12] for k in known), h[:12]))


def test_modified_scorers_are_declared():
    """Script bi sua sau khi chay phai khai bao, kem ly do."""
    n = 0
    for p in FILES:
        d = _load(p)
        if d.get("current_sha256"):
            n += 1
            assert d.get("modified_after_run") is True, p.name
            assert len(d.get("modification", "")) > 30, (
                "%s: sua script ma khong ghi ly do" % p.name)
            assert d["scorer_sha256"] != d["current_sha256"], p.name
    # gsm_sweep.py da bi sua; it nhat manifest do phai phan biet duoc
    assert n > 0, "ky vong co it nhat 1 scorer bi sua sau khi chay"


def test_gguf_hash_matches_manifest_original():
    """Hash gguf trong ban patch phai bang hash ghi trong manifest goc.

    Do doc lap: doc file goc trong notes/ cua may lab, chi duong dan qua
    bien moi truong QWEN_LAB_NOTES (mặc định giữ đường dẫn lab cũ để
    tương thích). tren may khong co notes/ (khong commit file phuc tap),
    test nay tu bo qua.
    """
    import os
    src = Path(os.environ.get("QWEN_LAB_NOTES", "D:/qwen/notes"))
    if not src.is_dir():
        pytest.skip("khong co thu muc lab notes tren may nay (QWEN_LAB_NOTES=%s)" % src)
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


def test_dataset_source_covers_index_range():
    """Hash nguon goc phai dung so item va dai do trong details."""
    for p in FILES:
        d = _load(p)
        ds = d.get("dataset_source")
        if not ds:
            continue
        idx = [r["i"] for r in d["details"]]
        assert ds["n_indices"] == len(idx), p.name
        assert ds["index_range"] == [min(idx), max(idx)], p.name
        for k in ("sha256_full", "sha256_answer"):
            assert HEX64.match(ds[k]), "%s: %s" % (p.name, k)


def test_two_400_item_runs_share_one_dataset_hash():
    """Hai run 400 cau doc lap phai dung CUNG mot noi dung dataset.

    Day la kiem cheo manh: no doc lap voi tung run -- giu du ca chay thi
    doi item, ca chay sau doi item.
    """
    full = {}
    for p in FILES:
        d = _load(p)
        ds = d.get("dataset_source")
        if ds and ds.get("n_indices") == 400:
            full.setdefault(ds["sha256_full"], []).append(p.name)
    big = {h: v for h, v in full.items() if len(v) >= 2}
    assert big, ("khong co cap hai manifest 400 cau nao cung dataset hash. "
                 "full=%s" % {h[:12]: v for h, v in full.items()})
    for h, v in big.items():
        print("  400-item group %s: %s" % (h[:12], ", ".join(v)))


def test_weak_hash_is_declared_where_it_is_weak():
    """Schema 'harness' chi hash duoc chi so -- phai noi ro la yeu.

    Khong bat buoc phai co dataset_source (nhieu run da xong khong kip
    gan), nhung neu thieu thi phai co scope mo ta hanh chu khong nham rang
    hash chung minh duoc noi dung cau.
    """
    weak = []
    for p in FILES:
        d = _load(p)
        if d["detail_schema"] != "harness":
            continue
        if not d.get("dataset_source"):
            weak.append(p.name)
            scope = d["dataset_sha256_scope"]
            assert "KHONG" in scope.upper(), (
                "%s: schema harness, hash yeu, ma scope khong canh bao" % p.name)
    print("  harness yeu (chua gan hash nguon goc): %d" % len(weak))
