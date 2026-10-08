"""Unified cross-scale free-response: canonical typed matching, N reps,
full raw outputs and provenance. The generation prompt is unchanged from V3.1;
new pass labels use the V3.3 scorer, not the historical V3.1 matcher.
Usage:
  python reproduce/eval_v3b.py --model <hf-dir> --questions data/v2/questions.json
      --out out.json --reps 3 --tag s30-05B
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

if __package__:
    from .evaluation_io import positive_int, validate_questions, write_json
    from .scoring import match
else:
    from evaluation_io import positive_int, validate_questions, write_json
    from scoring import match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=positive_int, default=3)
    ap.add_argument("--max-tokens", type=positive_int, default=30)
    a = ap.parse_args()

    qraw = Path(a.questions).read_bytes()
    Qs = validate_questions(json.loads(qraw), typed=True)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tok or a.model)
    m = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, device_map="cuda",
        low_cpu_mem_usage=True).eval()
    rep_scores, rep_items = [], []
    with torch.no_grad():
        for rep in range(a.reps):
            ok, items = 0, []
            for q in Qs:
                body = tok.apply_chat_template(
                    [{"role": "user",
                      "content": f"{q['question']} Answer with only the value, no explanation."}],
                    tokenize=False, add_generation_prompt=True)
                ids = tok(body, return_tensors="pt", add_special_tokens=False).to("cuda")
                out = m.generate(**ids, max_new_tokens=a.max_tokens,
                                 do_sample=False,
                                 eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.eos_token_id)
                t = tok.decode(out[0][ids.input_ids.shape[1]:])
                good = match(q, t)
                ok += good
                items.append({"id": q["id"], "pass": good, "out": t})
            rep_scores.append(ok)
            rep_items.append(items)
            print(f"rep {rep + 1}: {ok}/{len(Qs)}", flush=True)
    res = {"run_id": f"V33-{a.tag}-{time.strftime('%Y%m%d-%H%M%S')}",
           "model": a.model,
           "tokenizer": a.tok or a.model,
           "protocol": "v3.3-typed-final-answer",
           "generation_protocol": "v3.1-greedy-chat-template",
           "script_sha256": hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest()[:12],
           "scorer": "reproduce/scoring.py",
           "scorer_sha256": hashlib.sha256(
               Path(__file__).with_name("scoring.py").read_bytes()).hexdigest()[:12],
           "dataset": a.questions,
           "dataset_sha256": hashlib.sha256(qraw).hexdigest()[:12],
           "torch": torch.__version__, "transformers": transformers.__version__,
           "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0),
           "max_new_tokens": a.max_tokens, "do_sample": False,
           "rep_scores": rep_scores, "rep_items": rep_items}
    write_json(a.out, res)
    print(f"saved {a.out}: {rep_scores}")


if __name__ == "__main__":
    main()
