"""Unit tests for reproduce/scoring.py (run: pytest tests/ -q)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reproduce"))
from scoring import match
import pytest


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


@pytest.mark.parametrize("kind", ["unordered_numeric_set", "ordered_numeric_tuple"])
def test_fraction_collections_use_the_same_parser_for_keys_and_outputs(kind):
    q = Q(kind, ["(1/2, 3/4)"])
    assert match(q, "(0.5, 0.75)")
    assert match(q, "(1/2, 3/4)")
    assert not match(q, "(1, 2, 3, 4)")


@pytest.mark.parametrize("kind", ["unordered_numeric_set", "ordered_numeric_tuple"])
def test_invalid_collection_elements_are_not_silently_discarded(kind):
    q = Q(kind, ["2,3"])
    assert not match(q, "2,3,1/0")
    assert not match(q, "2,3x")


def test_scalar_marker_does_not_bypass_clean_number_check():
    q = Q("numeric_scalar", ["0.5"])
    assert not match(q, "#### 0.5x")
    assert match(q, "#### 0.5<|im_end|>")


def test_unordered_roots_preserve_multiplicity():
    q = Q("unordered_numeric_set", ["1,1"])
    assert match(q, "1,1")
    assert not match(q, "1")
