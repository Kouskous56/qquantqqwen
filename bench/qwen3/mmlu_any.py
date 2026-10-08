"""MMLU with the published round-robin subject sample and balanced labels."""
import argparse
from pathlib import Path

if __package__:
    from .common import (add_ollama_args, atomic_json, evaluate, mmlu_items,
                         parse_mc, safe_tag, sha256)
else:
    from common import (add_ollama_args, atomic_json, evaluate, mmlu_items,
                        parse_mc, safe_tag, sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True, type=safe_tag)
    parser.add_argument("--gguf", default="")
    parser.add_argument("--precision", required=True)
    add_ollama_args(parser)
    args = parser.parse_args(argv)
    import datasets
    items = mmlu_items(datasets.load_dataset("cais/mmlu", "all", split="test"))
    provenance = {}
    if args.gguf:
        path = Path(args.gguf)
        provenance = {"gguf": path.name, "gguf_sha256": sha256(path),
                      "gguf_bytes": path.stat().st_size}
    checkpoint = Path(args.notes_dir) / f"R2CKPT_MMLU_{args.tag}.json"
    output = Path(args.notes_dir) / f"RUN_MMLU200_{args.tag}.json"
    details, identity, elapsed = evaluate(args, items, "mmlu", checkpoint, __file__, 32,
                                         precision=args.precision, provenance=provenance)
    score = sum(row["pass"] for row in details)
    atomic_json(output, {"model": args.model, "tag": args.tag, "precision": args.precision,
                         "provenance": provenance, "n": len(details), "mmlu": score,
                         "acc": round(score / len(details), 4),
                         "n_subjects": len({row["subj"] for row in details}),
                         "unparsed": sum(row["unparsed"] for row in details),
                         "protocol": "ollama-greedy-MC-one-letter-npredict32",
                         "run_identity": identity, "details": details,
                         "elapsed_s": round(elapsed)})
    print(f"MMLU-{len(details)}: {score}/{len(details)} saved {output}")


if __name__ == "__main__":
    main()
