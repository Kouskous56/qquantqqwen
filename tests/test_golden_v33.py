"""Golden regression: rescore must reproduce canonical V3.3 numbers exactly."""
import json
import subprocess
import sys
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXPECTED = {
    "05B": {"s0": 18, "s30": 11, "n11": 8, "n10": 10, "n01": 3, "n00": 29},
    "15B": {"s0": 40, "s30": 33, "n11": 33, "n10": 7, "n01": 0, "n00": 10},
    "3B": {"s0": 42, "s30": 40, "n11": 39, "n10": 3, "n01": 1, "n00": 7},
}


def test_golden_v33():
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "reproduce", "rescore_v3.py"),
         "--manifest-dir", os.path.join(ROOT, "manifests"),
         "--questions", os.path.join(ROOT, "data", "v2", "questions.json"),
         "--out-results", os.path.join(ROOT, "results", "v3_cross_scale.csv"),
         "--out-manifest", os.path.join(ROOT, "manifests", "RUN_V33_RESCORE.json")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    got = {}
    for ln in open(os.path.join(ROOT, "results", "v3_cross_scale.csv"),
                   encoding="utf-8").read().strip().split("\n")[1:]:
        p = ln.split(",")
        got[p[0]] = {"s0": int(p[1]), "s30": int(p[2]), "n11": int(p[5]),
                     "n10": int(p[6]), "n01": int(p[7]), "n00": int(p[8])}
    assert got == EXPECTED, f"drift: {got}"
