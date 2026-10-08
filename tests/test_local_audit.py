import json
from pathlib import Path

import pytest

from reproduce.audit_local_data import inventory, verify_models
from reproduce.compare_gsm_runs import compare, load_blocks


def run(tmp_path, name, answers, *, offset=0, model="test-model"):
    path = tmp_path / name
    details = [{"i": offset + i, "exp": "42", "got": "42" if passed else "0", "pass": passed}
               for i, passed in enumerate(answers)]
    path.write_text(json.dumps({"model": model, "n": len(details), "offset": offset,
                               "gsm": sum(answers), "details": details}), encoding="utf-8")
    return path


def test_pairing_by_id_not_record_order(tmp_path):
    a = run(tmp_path, "a.json", [True, True, False, False])
    b = run(tmp_path, "b.json", [True, False, True, False])
    data = json.loads(b.read_text())
    data["details"].reverse()
    b.write_text(json.dumps(data))
    result = compare([a], [b])
    assert [result[k] for k in ("n11", "n10", "n01", "n00")] == [1, 1, 1, 1]
    assert result["mcnemar_exact_two_sided_p"] == 1.0


def test_exact_mcnemar_reference(tmp_path):
    a = run(tmp_path, "a.json", [True] * 7)
    b = run(tmp_path, "b.json", [False] * 7)
    assert compare([a], [b])["mcnemar_exact_two_sided_p"] == 0.015625


def test_reject_overlap_and_model_mix(tmp_path):
    a = run(tmp_path, "a.json", [True])
    with pytest.raises(ValueError, match="duplicate"):
        load_blocks([a, a])
    b = run(tmp_path, "b.json", [True], offset=1, model="other-model")
    with pytest.raises(ValueError, match="same model"):
        load_blocks([a, b])


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d.update(n=3), "incomplete"),
    (lambda d: d["details"][0].update({"pass": "false"}), "boolean"),
    (lambda d: d["details"][0].update({"got": "0"}), "inconsistent"),
    (lambda d: d.update(offset=100), "range"),
    (lambda d: d.update(gsm=0), "aggregate"),
])
def test_reject_invalid_runs(tmp_path, mutation, message):
    path = run(tmp_path, "a.json", [True])
    data = json.loads(path.read_text())
    mutation(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_blocks([path])


def test_reject_unpaired_questions(tmp_path):
    a = run(tmp_path, "a.json", [True])
    b = run(tmp_path, "b.json", [True], offset=1)
    with pytest.raises(ValueError, match="same question"):
        compare([a], [b])


def test_inventory_hashes_and_flags_corrupt_json(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "bad.json").write_text("{invalid")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.gguf").write_bytes(b"test")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "ignored.txt").write_text("excluded")
    records = inventory(tmp_path)
    assert len(records) == 2
    bypath = {r["path"]: r for r in records}
    assert "sha256" not in bypath["models/model.gguf"]
    assert bypath["notes/bad.json"]["json_valid"] is False
    hashed = inventory(tmp_path, hash_models=True)
    assert all(len(r["sha256"]) == 64 for r in hashed)
    assert (tmp_path / "notes" / "bad.json").read_text() == "{invalid"


def test_model_verification_distinguishes_unhashed_missing_and_match(tmp_path):
    records = [{"path": "models/a.gguf", "sha256": "a" * 64}, {"path": "models/b.gguf"}]
    for name in "abc":
        (tmp_path / f"RUN_{name}.json").write_text(json.dumps({
            "provenance": {"gguf": f"D:\\qwen\\models\\{name}.gguf", "gguf_sha256": name * 64}}))
    checks = verify_models(records, tmp_path)
    assert [r["available_now"] for r in checks] == [True, True, False]
    assert [r["hash_matches"] for r in checks] == [True, None, None]
