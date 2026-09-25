"""Download Qwen2.5-Instruct weights + verify sizes. Usage:
  python reproduce/download_weights.py --out <dir> [--scale 0.5B|1.5B|3B|all]
Requires: huggingface_hub. Models total ~10GB (0.9/2.9/5.9GB)."""
import argparse
import os

from huggingface_hub import snapshot_download

WEIGHTS = {
    "0.5B": ("Qwen/Qwen2.5-0.5B-Instruct", 942),
    "1.5B": ("Qwen/Qwen2.5-1.5B-Instruct", 2944),
    "3B": ("Qwen/Qwen2.5-3B-Instruct", 5897),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", default="all")
    a = ap.parse_args()
    scales = list(WEIGHTS) if a.scale == "all" else [a.scale]
    for s in scales:
        repo, expect_mb = WEIGHTS[s]
        d = snapshot_download(repo, local_dir=os.path.join(a.out, f"Qwen2.5-{s}"),
                              local_dir_use_symlinks=False)
        total = sum(os.path.getsize(os.path.join(r, f)) / 1e6
                    for r, _, fs in os.walk(d) for f in fs
                    if f.endswith(".safetensors"))
        print(f"{s}: {total:.0f}MB (expected ~{expect_mb}MB safetensors) in {d}")


main()
