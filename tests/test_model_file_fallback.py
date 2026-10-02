"""Hoi quy cho anh xa model -> GGUF du phong (tier 3 cua resolve_gguf).

Boi canh 2026-10-02: 24 manifest duoc giai quyet qua blob Ollama
(`ollama-modelfile-FROM-blob`); blob co the mat khi don dep. File
`manifests_qwen3/_MODEL_FILE_FALLBACK.json` la nguon duy nhat cho anh xa
nay. Test nay KHONG can GPU/dia/model that — chi kiem tinh nhat quan
giua map va manifest da phat hanh:

  1. moi model tung duoc giai qua blob deu co mat trong map;
  2. ten file trong map trung provenance.gguf cua manifest do
     (map sai thi patcher chay lai se gan sai file);
  3. moi dong trong map deu co it nhat 1 manifest tham chieu
     (chong muc chet khong ai dung).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manifests_qwen3"
MAP = MAN / "_MODEL_FILE_FALLBACK.json"


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _manifests():
    out = []
    for p in sorted(MAN.glob("RUN_*.json")):
        d = _load(p)
        out.append((p.name, d.get("model"),
                    (d.get("provenance") or {}).get("gguf"),
                    (d.get("provenance") or {}).get("gguf_resolved_via")))
    return out


def _map():
    d = _load(MAP)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def test_fallback_file_hop_le():
    m = _map()
    assert len(m) >= 13, "map thieu muc: %d" % len(m)
    for k, v in m.items():
        assert k and isinstance(k, str), "key rong"
        assert v.endswith(".gguf"), "%s -> %s khong phai GGUF" % (k, v)
    print("  fallback map: %d model" % len(m))


def test_blob_resolved_models_duoc_bao_phu():
    m = _map()
    thieu = [(n, mo) for n, mo, g, v in _manifests()
             if v == "ollama-modelfile-FROM-blob" and mo not in m]
    assert not thieu, "model giai qua blob nhung khong co trong map: %s" % thieu
    print("  blob-resolved: phu het")


def test_map_trung_provenance_da_phat_hanh():
    m = _map()
    lech = [(n, mo, m[mo], g) for n, mo, g, v in _manifests()
            if v == "ollama-modelfile-FROM-blob" and m.get(mo) != g]
    assert not lech, "map lech provenance: %s" % lech
    print("  map trung provenance tai moi manifest blob-resolved")


def test_khong_muc_chet():
    m = _map()
    models = {mo for _, mo, _, _ in _manifests()}
    chet = [k for k in m if k not in models]
    assert not chet, "muc khong manifest nao tham chieu: %s" % chet
    print("  khong muc chet")
