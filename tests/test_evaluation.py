"""CPU regressions for evaluation integrity and dependency-free CLI help."""
import copy
import hashlib
import importlib
import json
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce.evaluation_io import validate_questions
from reproduce.eval_gsm import extract, extract_numeric
from reproduce.rescore_v3 import mcnemar_exact, score_repetitions


def question(qid="q1"):
    return {"id": qid, "question": "Midpoint?", "answer_type": "ordered_numeric_tuple",
            "accepted": ["(2,3)"], "choices": ["(2,3)", "(3,2)", "(0,0)", "(1,1)"],
            "answer": "A"}


def repetitions():
    return {"rep_items": [[{"id": "q1", "out": "(2,3)", "pass": False}] for _ in range(3)]}


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d["rep_items"][0].append({"id": "q1", "out": "(3,2)"}), "duplicate"),
    (lambda d: d["rep_items"][1].clear(), "missing"),
    (lambda d: d["rep_items"][2][0].update(id="unknown"), "unknown"),
    (lambda d: d["rep_items"][0][0].update(out=None), "out must be text"),
    (lambda d: d["rep_items"].pop(), "exactly three"),
    (lambda d: d["rep_items"].append([]), "exactly three"),
])
def test_rescore_rejects_invalid_pairing(mutation, message):
    data = repetitions()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        score_repetitions(data, {"q1": question()}, "run.json")


def test_rescore_uses_raws_not_historical_labels():
    assert score_repetitions(repetitions(), {"q1": question()}, "run.json") == [
        {"q1": True}, {"q1": True}, {"q1": True}
    ]


@pytest.mark.parametrize("questions, message", [
    ([], "non-empty"),
    ([question(), question()], "duplicate"),
    ([dict(question(), accepted=[])], "accepted"),
    ([dict(question(), answer_type="typo")], "answer_type"),
])
def test_question_validation(questions, message):
    with pytest.raises(ValueError, match=message):
        validate_questions(questions, typed=True)


def test_mcnemar_known_values_and_symmetry():
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(7, 0) == pytest.approx(0.015625)
    assert mcnemar_exact(10, 3) == mcnemar_exact(3, 10)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


@pytest.mark.parametrize("module", [
    "eval_v2", "eval_v2_hf", "eval_v3", "eval_v3b", "eval_gsm", "rescore_v3",
])
def test_evaluation_import_and_help_without_ml_dependencies(module):
    # -S disables site-packages, so this fails if a CLI imports Torch, datasets,
    # transformers or llama_cpp before parsing --help, even on a GPU workstation.
    imported = subprocess.run(
        [sys.executable, "-S", "-c", f"import reproduce.{module}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert imported.returncode == 0, imported.stderr
    help_run = subprocess.run(
        [sys.executable, "-S", str(ROOT / "reproduce" / f"{module}.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert help_run.returncode == 0, help_run.stderr
    assert "usage:" in help_run.stdout


@pytest.mark.parametrize("text, expected", [
    ("The answer is 42.", "42"),
    ("#### 42.0", "42"),
    ("#### 1,234", "1234"),
    ("#### 1e3", "1000"),
    ("#### .5", "1/2"),
    (r"\boxed{0.5}", "1/2"),
    (r"\boxed{1/2}", "1/2"),
    (r"\boxed{12} Correction: \boxed{42}", "42"),
    ("#### 12\nCorrection: #### 42", "42"),
    ("#### -42", "-42"),
    ("#### 1/0", ""),
    ("#### 42x", ""),
    (r"Earlier 42, final \boxed{unknown}", ""),
    ("Earlier 42, final #### unknown", ""),
    ("No answer", ""),
])
def test_numeric_gsm_extractor(text, expected):
    assert extract_numeric(text) == expected


def test_gsm_historical_parser_is_available_unchanged():
    assert extract("The answer is 42.") == "42."
    assert extract(r"\boxed{12} Correction: \boxed{42}") == "12"


def test_v3_generation_scores_current_dataset_types_and_preserves_raws(tmp_path, monkeypatch):
    evaluator = importlib.import_module("reproduce.eval_v3b")
    questions = [question()]
    qpath = tmp_path / "questions.json"
    # A CRLF file verifies that provenance hashes the file bytes on Windows.
    qbytes = json.dumps(questions, indent=1).replace("\n", "\r\n").encode()
    qpath.write_bytes(qbytes)
    output = " " * 350 + "(2,3)<|im_end|>"

    class Tokens(dict):
        input_ids = SimpleNamespace(shape=(1, 0))

        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            return Tokens()

        def decode(self, value):
            return value

    class Model:
        def eval(self):
            return self

        def generate(self, **kwargs):
            return [output]

    torch = SimpleNamespace(
        __version__="test", float16="test", no_grad=nullcontext,
        version=SimpleNamespace(cuda="test"),
        cuda=SimpleNamespace(get_device_name=lambda _: "test"),
    )
    transformers = SimpleNamespace(
        __version__="test", AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model()),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    out = tmp_path / "nested" / "run.json"
    monkeypatch.setattr(sys, "argv", ["eval_v3b", "--model", "fake", "--questions", str(qpath),
                                     "--out", str(out), "--tag", "test"])
    evaluator.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["rep_scores"] == [1, 1, 1]
    assert all(rep[0]["out"] == output for rep in data["rep_items"])
    assert data["dataset_sha256"] == hashlib.sha256(qbytes).hexdigest()[:12]
    assert data["protocol"] == "v3.3-typed-final-answer"
    assert data["scorer_sha256"] == hashlib.sha256(
        (ROOT / "reproduce" / "scoring.py").read_bytes()
    ).hexdigest()[:12]
