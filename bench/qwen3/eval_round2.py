"""Evaluate one GGUF with MMLU-200 and two independent GSM-200 blocks."""
import argparse
from pathlib import Path

if __package__:
    from .common import (add_ollama_args, atomic_json, evaluate, extract,
                         gsm_items, mmlu_items, parse_mc, safe_tag, sha256)
else:
    from common import (add_ollama_args, atomic_json, evaluate, extract,
                        gsm_items, mmlu_items, parse_mc, safe_tag, sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True, type=safe_tag)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--imatrix", default="unknown")
    parser.add_argument("--llama-cpp", default="unknown", help="actual runtime/build identifier")
    parser.add_argument("--questions", default="", help="deprecated; these benchmarks use HF datasets")
    add_ollama_args(parser)
    args = parser.parse_args(argv)
    if args.questions:
        parser.error("--questions is unused by MMLU/GSM evaluation; remove it")
    path = Path(args.gguf)
    provenance = {"gguf": path.name, "gguf_sha256": sha256(path),
                  "gguf_bytes": path.stat().st_size, "imatrix": args.imatrix,
                  "llama_cpp": args.llama_cpp}
    notes = Path(args.notes_dir)
    import datasets
    items = mmlu_items(datasets.load_dataset("cais/mmlu", "all", split="test"))
    details, identity, elapsed = evaluate(
        args, items, "mmlu", notes / f"R2CKPT_MMLU_{args.tag}.json", __file__, 32,
        provenance=provenance)
    score = sum(row["pass"] for row in details)
    atomic_json(notes / f"RUN_QWEN3_R2_MMLU200_{args.tag}.json", {
        "model": args.model, "tag": f"R2_MMLU200_{args.tag}", "provenance": provenance,
        "n": len(details), "mmlu": score, "acc": round(score / len(details), 4),
        "n_subjects": len({row["subj"] for row in details}),
        "unparsed": sum(row["unparsed"] for row in details),
        "protocol": "ollama-greedy-MC-one-letter-npredict32", "run_identity": identity,
        "details": details, "elapsed_s": round(elapsed)})
    dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")
    # Validate both blocks before any GSM request.
    blocks = [gsm_items(dataset, offset, 200) for offset in (0, 200)]
    all_details, identities, total_elapsed = [], [], 0.0
    for offset, items in zip((0, 200), blocks):
        details, identity, elapsed = evaluate(
            args, items, "gsm", notes / f"R2CKPT_GSM_{args.tag}_{offset}.json",
            __file__, 256, provenance=provenance)
        all_details.extend(details)
        identities.append(identity)
        total_elapsed += elapsed
    score = sum(row["pass"] for row in all_details)
    atomic_json(notes / f"RUN_QWEN3_R2_GSM400_{args.tag}.json", {
        "model": args.model, "tag": f"R2_GSM400_{args.tag}", "provenance": provenance,
        "n": len(all_details), "gsm": score, "gsm400": score,
        "blockA": sum(row["pass"] for row in all_details if row["i"] < 200),
        "blockB": sum(row["pass"] for row in all_details if row["i"] >= 200),
        "protocol": "ollama-greedy-CoT-####-npredict256", "run_identities": identities,
        "details": all_details, "elapsed_s": round(total_elapsed)})
    print(f"GSM-400: {score}/{len(all_details)}; DONE {args.tag}")


if __name__ == "__main__":
    main()
