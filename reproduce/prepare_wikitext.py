"""Deterministic WikiText test-corpus builder (first 200 rows, drop empties).
Usage: python reproduce/prepare_wikitext.py --out bench-corpus/wikitext_test.txt
Records: dataset Salesforce/wikitext wikitext-2-raw-v1 test split,
first 200 rows with empty rows removed, SHA256 of output bytes.
"""
import argparse
import hashlib
from pathlib import Path


def build_corpus(rows):
    """Preserve the historical selection rule and emit platform-independent UTF-8."""
    return "\n".join(t for t in rows[:200] if t.strip()).encode("utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    # EXACT historical rule (do not "improve"): first 200 rows, drop empties
    blob = build_corpus(ds["text"])
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    print("rows:", sum(bool(t.strip()) for t in ds["text"][:200]))
    print("sha256:", hashlib.sha256(blob).hexdigest())
    print("saved", a.out)


if __name__ == "__main__":
    main()
