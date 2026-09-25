"""Deterministic WikiText test-corpus builder (200 non-empty lines).
Usage: python reproduce/prepare_wikitext.py --out bench-corpus/wikitext_test.txt
Records: dataset Salesforce/wikitext wikitext-2-raw-v1 test split,
first 200 non-empty lines, SHA256 of output.
"""
import argparse
import hashlib

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    # EXACT historical rule (do not "improve"): first 200 rows, drop empties
    lines = [t for t in ds["text"][:200] if t.strip()]
    blob = "\n".join(lines)
    open(a.out, "w", encoding="utf-8").write(blob)
    print("lines:", len(lines))
    print("sha256:", hashlib.sha256(blob.encode()).hexdigest())
    print("saved", a.out)


main()
