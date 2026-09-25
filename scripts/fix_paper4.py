import re
R = "D:/qwen/report/"
p = open(R + "paper.tex", encoding="utf-8").read()
m = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", p, re.S)
assert m, "inline abstract missing"
p = p[:m.start()] + "\\input{abstract}" + p[m.end():]
open(R + "paper.tex", "w", encoding="utf-8").write(p)

q = open(R + "sec_quant_method.tex", encoding="utf-8").read()
old = ("with permutation leaving ARC/HellaSwag invariant under likelihood scoring---confirming\n"
       "that our generative bias finding is protocol-specific, not a data artifact.")
assert old in q, "metabench wording missing"
q = q.replace(old, "with permutation leaving ARC/HellaSwag scores essentially unchanged, "
              "showing that likelihood-scored multiple choice and generative label "
              "selection expose different failure modes.")
open(R + "sec_quant_method.tex", "w", encoding="utf-8").write(q)

t = open(R + "sec_scale_deploy.tex", encoding="utf-8").read()
old2 = "Hybrid s30+Q4 keeps 94--97\% instruction following (IFEval strict 0.601 $\\rightarrow$ 0.566)."
assert old2 in t, "hybrid para missing"
t = t.replace(old2, "Hybrid s30+Q4 keeps 94--97\% instruction following (IFEval strict "
              "0.601 $\\rightarrow$ 0.566), measured after the paragraph to avoid "
              "splitting prose across the deployment table.")
open(R + "sec_scale_deploy.tex", "w", encoding="utf-8").write(t)
print("all three fixed")
