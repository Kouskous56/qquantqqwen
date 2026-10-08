"""Standardized PPL for historical checkpoint names, using corrected token counts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce.eval_ppl import PROTOCOL, evaluate_perplexity

B = "D:/qwen/models/"
MODEL_NAMES = [
    ("05B-s0", "Qwen2.5-0.5B-FP16", "Qwen2.5-0.5B-FP16"),
    ("05B-s30", "Qwen2.5-0.5B-pruned-s30", "Qwen2.5-0.5B-FP16"),
    ("15B-s0", "Qwen2.5-1.5B-FP16", "Qwen2.5-1.5B-FP16"),
    ("15B-s30", "Qwen2.5-1.5B-FP16-pruned-s30", "Qwen2.5-1.5B-FP16"),
    ("3B-s0", "Qwen2.5-3B-FP16", "Qwen2.5-3B-FP16"),
    ("3B-s30", "Qwen2.5-3B-FP16-pruned-s30", "Qwen2.5-3B-FP16"),
    ("3B-s40", "Qwen2.5-3B-FP16-pruned-s40", "Qwen2.5-3B-FP16"),
    ("3B-s50", "Qwen2.5-3B-FP16-pruned-s50", "Qwen2.5-3B-FP16"),
    ("3B-sgpt24", "Qwen2.5-3B-sparsegpt-24", "Qwen2.5-3B-FP16"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=B)
    parser.add_argument("--corpus", default="D:/qwen/bench/wikitext_test.txt")
    parser.add_argument("--out", default="D:/qwen/notes/RUN_PPL_STD_V2.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    text = Path(args.corpus).read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    results = {}
    for name, model_name, tokenizer_name in MODEL_NAMES:
        started = time.time()
        tok = AutoTokenizer.from_pretrained(str(Path(args.models_dir) / tokenizer_name))
        model = AutoModelForCausalLM.from_pretrained(
            str(Path(args.models_dir) / model_name), dtype=dtype,
            device_map=args.device, low_cpu_mem_usage=True).eval()
        encoded = tok(text, return_tensors="pt")
        ppl, tokens = evaluate_perplexity(model, encoded.input_ids, device=args.device)
        results[name] = {"ppl": round(ppl, 2), "tokens": tokens,
                         "input_tokens": encoded.input_ids.shape[1],
                         "runtime_s": round(time.time() - started, 1)}
        print(f"{name}: PPL {ppl:.2f} ({tokens} predicted tokens)", flush=True)
        del model, tok
        if args.device == "cuda":
            torch.cuda.empty_cache()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": PROTOCOL, "corpus_sha": digest,
                                 "max_len": 512, "stride": 256,
                                 "dtype": str(dtype), "device": args.device,
                                 "results": results}, indent=1, allow_nan=False), encoding="utf-8")
    print("saved", output)


if __name__ == "__main__":
    main()
