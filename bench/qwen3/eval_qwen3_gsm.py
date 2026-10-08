"""FP16 Transformers GSM evaluation; imports and GPU loading occur only in main."""
import argparse
from pathlib import Path
import time

if __package__:
    from .common import atomic_json, extract, gold_answer, gsm_items, positive_int, sha256, GSM_PROMPT
else:
    from common import atomic_json, extract, gold_answer, gsm_items, positive_int, sha256, GSM_PROMPT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="D:/qwen/models/Qwen3-4B-FP16")
    parser.add_argument("--notes-dir", default="D:/qwen/notes")
    parser.add_argument("--n", type=positive_int, default=50)
    args = parser.parse_args(argv)
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    items = gsm_items(load_dataset("openai/gsm8k", "main", split="test"), 0, args.n)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_root, dtype=torch.float16, device_map="auto").eval()
    # Dispatch hooks handle offloaded layers. Do not assume a CUDA device exists.
    device = model.get_input_embeddings().weight.device
    if device.type == "meta":
        device = model.device
    if device.type == "meta":
        device = torch.device("cpu")
    started, details = time.monotonic(), []
    for index, row in items:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"] + GSM_PROMPT}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:])
        expected, got = gold_answer(row["answer"]), extract(text)
        details.append({"i": index, "pass": got == expected, "exp": expected,
                        "got": got, "text": text})
        print(f"gsm {index}: {got == expected}", flush=True)
    score = sum(row["pass"] for row in details)
    output = Path(args.notes_dir) / "RUN_QWEN3_GSM.json"
    atomic_json(output, {"model": args.model_root, "gsm50_256": score, "gsm": score,
                         "n": len(details), "details": details,
                         "protocol": "transformers-greedy-CoT-####-max_new_tokens256",
                         "scorer_sha256": sha256(__file__),
                         "common_sha256": sha256(Path(__file__).with_name("common.py")),
                         "elapsed_s": round(time.monotonic() - started),
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"GSM-{len(details)}: {score}/{len(details)} saved {output}")


if __name__ == "__main__":
    main()
