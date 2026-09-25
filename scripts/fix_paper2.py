import re
R = "D:/qwen/report/"

# 1. Split sec_intro_env -> sec_intro + sec_env
t = open(R + "sec_intro_env.tex", encoding="utf-8").read()
parts = t.split("\\section{Experimental Environment}")
assert len(parts) == 2
open(R + "sec_intro.tex", "w", encoding="utf-8").write(parts[0])
open(R + "sec_env.tex", "w", encoding="utf-8").write(
    "\\section{Experimental Environment}" + parts[1])

# 2. Related Work expanded, integrated citations
related = """\\section{Related Work}
Post-training quantization via GPTQ~\\cite{gptq} and AWQ~\\cite{awq} showed that
LLMs compress to 4 bits with little loss; we test whether that optimism survives
behavioral (not just loss) evaluation on Qwen2.5~\\cite{qwen2} at three scales.
One-shot pruning via SparseGPT~\\cite{sparsegpt} and Wanda~\\cite{wanda} motivates
our sparsity axis, including the 2:4 pattern we evaluate against unstructured masks.
For recovery we build on LoRA~\\cite{lora} with inference-time mask reapplication.
Evaluation follows the EleutherAI harness~\\cite{lmeval} (MetaBench, WikiText) and
adds IFEval~\\cite{ifeval} plus GSM8K~\\cite{gsm8k} generation probes, because our
early results showed loss metrics missing generative failures.
Our contribution is not a new compressor but the cross-scale, behavior-first
comparison under fixed edge hardware.
"""
open(R + "sec_related.tex", "w", encoding="utf-8").write(related)

# 3. Remove Related Work from sec_scale_deploy
s = open(R + "sec_scale_deploy.tex", encoding="utf-8").read()
start = s.find("\\section{Related Work}")
end = s.find("\\section{Edge Deployment")
assert start > 0 and end > start
s = s[:start] + s[end:]

# 4. Conclusion contradiction fix + regime naming
s = s.replace(
"""quantize to Q4 first (large, safe gains), prune unstructured only within the
scale-specific safe regime ($\\sim$20\\% at 0.5B, $\\sim$30--40\\% at 3B), validate
with generation tests rather than perplexity alone, and do not expect sparse
weights to accelerate dense runtimes.""",
"""quantize to Q4 first (large, safe gains). Do not apply unstructured pruning solely
for deployment efficiency unless the runtime and storage stack can exploit sparsity:
on dense runtimes it saves nothing. If pruning is used, validate the chosen level
behaviorally rather than relying on perplexity alone; in our data s30 is a robust
regime, s40 a moderate-degradation regime, and s50 a collapse region at 3B.""")

# 5. Vietnamese refusal soften
s = s.replace("0.5B is 3$\\times$ faster but refuses Vietnamese prompts.",
 "0.5B is 3$\\times$ faster but showed substantially poorer Vietnamese instruction "
 "behavior in the 8-prompt demo, including a refusal-like failure on one prompt.")

# 6. GSM limitation wording
s = s.replace(
"""train and test share distribution, so claims cover recovery rather
than generalization; 2:4 stops at ns64 per a preregistered gate.""",
"""GSM8K recovery is evaluated on held-out samples from the same task distribution;
it demonstrates in-distribution generalization, not cross-domain recovery. 2:4 stops
at ns64 per a preregistered gate.""")

# 7. MetaBench wording soften
s = s.replace(
"""permutation leaves
ARC/HellaSwag invariant---likelihood scoring conditions on each candidate, unlike
generative label selection.""",
"""permutation leaves
ARC/HellaSwag scores essentially unchanged, showing that likelihood-scored multiple
choice and generative label selection expose different failure modes.""")

open(R + "sec_scale_deploy.tex", "w", encoding="utf-8").write(s)

# 8. paper.tex input order
p = open(R + "paper.tex", encoding="utf-8").read()
p = p.replace("\\input{sec_intro_env}",
              "\\input{sec_intro}\n\\input{sec_related}\n\\input{sec_env}")
open(R + "paper.tex", "w", encoding="utf-8").write(p)
print("restructure done")
