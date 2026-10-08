"""Token-weighted causal PPL with each target scored exactly once.

Historical PPL artifacts used an earlier window/counting rule. New results carry
an explicit protocol identifier and must not silently replace those artifacts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

PROTOCOL = "causal-token-weighted-v2"


def iter_windows(seq_len, max_len=512, stride=256):
    """Yield (context start, end, first target), in absolute token positions."""
    if max_len < 2 or not 0 < stride < max_len:
        raise ValueError("require max_len >= 2 and 0 < stride < max_len")
    if seq_len < 2:
        raise ValueError("perplexity requires at least two tokens")
    previous_end = 1  # The first token has no causal prediction to score.
    for end in range(stride, seq_len + stride, stride):
        end = min(end, seq_len)
        begin = max(end - max_len, 0)
        if end > previous_end:
            yield begin, end, previous_end
        previous_end = end


def evaluate_perplexity(model, input_ids, max_len=512, stride=256, device=None):
    """Evaluate one tokenized stream, accounting for the model's label shift."""
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("expected input_ids with shape [1, sequence_length]")
    if device is None:
        device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    with torch.inference_mode():
        for begin, end, target_start in iter_windows(
                input_ids.shape[1], max_len, stride):
            ids = input_ids[:, begin:end].to(device)
            labels = ids.clone()
            labels[:, :target_start - begin] = -100
            count = int((labels[:, 1:] != -100).sum().item())
            loss = float(model(ids, labels=labels, use_cache=False).loss.item())
            if not math.isfinite(loss):
                raise ValueError(f"non-finite loss in token window {begin}:{end}")
            total_nll += loss * count
            total_tokens += count
    try:
        perplexity = math.exp(total_nll / total_tokens)
    except OverflowError as exc:
        raise ValueError("perplexity overflow; model loss is too large") from exc
    return perplexity, total_tokens


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = ap.parse_args(argv)
    if a.max_len < 2 or not 0 < a.stride < a.max_len:
        ap.error("require --max-len >= 2 and 0 < --stride < --max-len")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = Path(a.corpus).read_text(encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(a.tok)
    t0 = time.time()
    dtype = torch.float16 if a.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, device_map=a.device,
        low_cpu_mem_usage=True).eval()
    enc = tok(text, return_tensors="pt")
    ppl, nt = evaluate_perplexity(model, enc.input_ids, a.max_len, a.stride,
                                 a.device)
    res = {"model": a.model, "corpus_sha": hashlib.sha256(text.encode()).hexdigest()[:12],
           "protocol": PROTOCOL, "max_len": a.max_len, "stride": a.stride,
           "input_tokens": enc.input_ids.shape[1], "tokens": nt,
           "ppl": round(ppl, 2), "runtime_s": round(time.time() - t0, 1),
           "torch": torch.__version__, "dtype": str(dtype), "device": a.device}
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(res, indent=1, allow_nan=False), encoding="utf-8")
    print(f"PPL {ppl:.2f} ({nt} predicted tokens)")


if __name__ == "__main__":
    main()
