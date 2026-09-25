"""Unit tests for reproduce/scoring.py (run: pytest tests/ -q)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reproduce"))
from scoring import match


def Q(t, acc):
    return {"answer_type": t, "accepted": acc}


def test_fraction():
    q = {"answer_type": "numeric_scalar", "accepted": ["0.5"]}
    assert match(q, "0.5<|im_end|>")
    assert match(q, "1/2")
    assert not match(q, "0.5x")


def test_unordered_roots():
    q = {"answer_type": "unordered_numeric_set", "accepted": ["2,3"]}
    assert match(q, "2, 3")
    assert match(q, "3, 2")
    assert not match(q, "2, 4")


def test_midpoint_order():
    q = {"answer_type": "ordered_numeric_tuple", "accepted": ["(2,3)"]}
    assert match(q, "(2, 3)")
    assert not match(q, "(3, 2)")


def test_symbolic_strict():
    q = {"answer_type": "symbolic_exact", "accepted": ["e^x+c"]}
    assert match(q, "e^x + C")
    assert not match(q, "ln(e^x) + C")


def test_symbolic_latex():
    q = {"answer_type": "symbolic_exact", "accepted": ["4pi"]}
    assert match(q, "$4\\pi$")
    assert match(q, "4π")
    assert not match(q, "12.56")


def test_discriminant_zero():
    q = {"answer_type": "numeric_scalar", "accepted": ["0"]}
    assert match(q, "0")
    assert not match(q, "0.5")
