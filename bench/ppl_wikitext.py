"""Single-checkpoint WikiText PPL with corrected causal token accounting."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce.eval_ppl import main as evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="D:/qwen/models/Qwen2.5-0.5B-FP16")
    parser.add_argument("--tok")
    parser.add_argument("--corpus", default="D:/qwen/bench/wikitext_test.txt")
    parser.add_argument("--out", default="D:/qwen/notes/PPL_BASELINE_V2.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args(argv)
    evaluate(["--model", args.model, "--tok", args.tok or args.model,
              "--corpus", args.corpus, "--out", args.out,
              "--device", args.device, "--max-len", "1024", "--stride", "512"])


if __name__ == "__main__":
    main()
