import argparse
import json

import pytest

from bench.qwen3 import common, patch_manifests


@pytest.mark.parametrize("text, expected", [("b", 1), ("The answer is c", 2), ("(D).", 3), ("unknown", -1)])
def test_mmlu_parser(text, expected):
    assert common.parse_mc(text, ["one", "two", "three", "four"]) == expected


def test_resume_after_interruption_and_reject_changed_run(tmp_path, monkeypatch):
    args = argparse.Namespace(model="test", endpoint="http://localhost:11434", tag="test")
    items = [(i, {"question": f"Question {i}", "answer": "#### 42"}) for i in range(3)]
    calls = []
    def interrupted(*params):
        calls.append(params)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return "#### 42"
    monkeypatch.setattr(common, "chat", interrupted)
    checkpoint = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="interrupted"):
        common.evaluate(args, items, "gsm", checkpoint, __file__, 256)
    assert json.loads(checkpoint.read_text())["done"] == 1
    calls.clear()
    monkeypatch.setattr(common, "chat", lambda *params: calls.append(params) or "#### 42")
    details, identity, _ = common.evaluate(args, items, "gsm", checkpoint, __file__, 256)
    assert len(calls) == 2
    assert len(details) == 3
    assert all(row["pass"] for row in details)
    args.model = "other"
    with pytest.raises(ValueError, match="different or legacy"):
        common.evaluate(args, items, "gsm", checkpoint, __file__, 256)
    args.model = "test"
    saved = json.loads(checkpoint.read_text())
    saved["details"][0]["pass"] = False
    checkpoint.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="score does not match"):
        common.load_checkpoint(checkpoint, identity, items, "gsm")


def test_corrupt_checkpoint_is_preserved(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{broken")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        common.load_checkpoint(path, {}, [], "gsm")
    assert path.read_text() == "{broken"


def test_atomic_json_preserves_previous_on_serialization_error(tmp_path):
    path = tmp_path / "a.json"
    common.atomic_json(path, {"value": 1})
    with pytest.raises(ValueError):
        common.atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"value": 1}
    assert len(list(tmp_path.iterdir())) == 1


def test_dataset_bounds_and_short_subjects():
    with pytest.raises(ValueError, match="exceeds"):
        common.gsm_items([{}], 0, 2)
    dataset = [{"subject": "a"}, {"subject": "b"}, {"subject": "b"}]
    assert [i for i, _ in common.mmlu_items(dataset, 3)] == [0, 1, 2]
    with pytest.raises(ValueError, match="contains"):
        common.mmlu_items(dataset, 4)


def test_fallback_map_rejects_traversal(tmp_path, monkeypatch):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"model": "../elsewhere.gguf"}))
    monkeypatch.setattr(patch_manifests, "FALLBACK_FILE", path)
    with pytest.raises(ValueError, match="Invalid GGUF"):
        patch_manifests.model_file_fallback()


def test_harness_shape_is_not_enough_to_attribute_script():
    scorer, _, _ = patch_manifests.attribute("RUN_UNKNOWN.json", {}, "harness", {"i": 0, "pass": True})
    assert scorer is None


def test_new_source_identity_tracks_common_dependency():
    sources = {"gsm_sweep.py": common.sha256(common.Path(common.__file__).with_name("gsm_sweep.py")),
               "common.py": common.sha256(common.__file__)}
    result = patch_manifests.recorded_scorer({"run_identity": {"sources": sources}})
    assert result["status"] == "verified-on-disk"
    assert result["dependencies"]["bench/qwen3/common.py"] == sources["common.py"]
