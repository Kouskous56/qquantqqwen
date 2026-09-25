"""Download Qwen2.5-Instruct weights + verify sizes. Usage:
  python reproduce/download_weights.py --out <dir> [--scale 0.5B|1.5B|3B|all]
Requires: huggingface_hub. Models total ~10GB (0.9/2.9/5.9GB)."""
import argparse
import os

from huggingface_hub import snapshot_download

WEIGHTS = {
    # local dir name: (upstream repo, upstream revision SHA, ~safetensors MB)
    "Qwen2.5-0.5B-FP16": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf", 942),
    "Qwen2.5-1.5B-FP16": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4c", 2944),
    "Qwen2.5-3B-FP16": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993", 5897),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", default="all")
    a = ap.parse_args()
    scales = [s for s in WEIGHTS if a.scale == "all" or s == a.scale
              or s.endswith(a.scale)]
    for s in scales:
        repo, rev, expect_mb = WEIGHTS[s]
        d = snapshot_download(repo, revision=rev,
                              local_dir=os.path.join(a.out, s),
                              local_dir_use_symlinks=False)
        total = sum(os.path.getsize(os.path.join(r, f)) / 1e6
                    for r, _, fs in os.walk(d) for f in fs
                    if f.endswith(".safetensors"))
        ok = abs(total - expect_mb) / expect_mb < 0.05
        print(f"{s}: {total:.0f}MB (expected ~{expect_mb}MB) "
              f"in {d} -> {'OK' if ok else 'SIZE MISMATCH'}")
        assert ok, f"size mismatch for {s}"


main()
