"""Download Qwen2.5-Instruct weights + verify sizes. Usage:
  python reproduce/download_weights.py --out <dir> [--scale 0.5B|1.5B|3B|all]
Requires: huggingface_hub. Models total ~10GB (0.9/2.9/5.9GB)."""
import argparse
import os

WEIGHTS = {
    # local dir name: (upstream repo, upstream revision SHA, ~safetensors MB)
    "Qwen2.5-0.5B-FP16": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf", 942),
    "Qwen2.5-1.5B-FP16": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4c", 2944),
    "Qwen2.5-3B-FP16": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993", 5897),
}


def select_scales(scale):
    """Accept the documented short scale or an exact local directory name."""
    if scale == "all":
        return list(WEIGHTS)
    name = f"Qwen2.5-{scale}-FP16"
    if scale in WEIGHTS:
        return [scale]
    if name in WEIGHTS:
        return [name]
    raise ValueError(f"unknown scale {scale!r}; choose 0.5B, 1.5B, 3B or all")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", default="all")
    a = ap.parse_args(argv)
    try:
        scales = select_scales(a.scale)
    except ValueError as exc:
        ap.error(str(exc))
    from huggingface_hub import snapshot_download
    for s in scales:
        repo, rev, expect_mb = WEIGHTS[s]
        d = snapshot_download(repo, revision=rev,
                              local_dir=os.path.join(a.out, s))
        total = sum(os.path.getsize(os.path.join(r, f)) / 1e6
                    for r, _, fs in os.walk(d) for f in fs
                    if f.endswith(".safetensors"))
        ok = abs(total - expect_mb) / expect_mb < 0.05
        print(f"{s}: {total:.0f}MB (expected ~{expect_mb}MB) "
              f"in {d} -> {'OK' if ok else 'SIZE MISMATCH'}")
        if not ok:
            raise RuntimeError(f"size mismatch for {s}: {total:.0f}MB, expected ~{expect_mb}MB")


if __name__ == "__main__":
    main()
