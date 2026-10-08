"""Historical Wanda CLI defaults backed by the corrected canonical implementation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = "D:/qwen/models/Qwen2.5-0.5B-FP16"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=SRC)
    parser.add_argument("--out")
    parser.add_argument("--sparsity", type=float, default=0.2)
    parser.add_argument("--semi24", action="store_true")
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=512)
    args = parser.parse_args(argv)
    tag = f"s{int(args.sparsity * 100)}{'-24' if args.semi24 else ''}"
    source = Path(args.src)
    output = args.out or str(source.parent / f"{source.name}-pruned-{tag}")
    from reproduce.prune_wanda import main as prune
    cli = ["--src", args.src, "--out", output, "--sparsity", str(args.sparsity),
           "--nsamples", str(args.nsamples), "--seqlen", str(args.seqlen)]
    if args.semi24:
        cli.append("--semi24")
    prune(cli)


if __name__ == "__main__":
    main()
