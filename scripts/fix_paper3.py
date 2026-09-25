R = "D:/qwen/report/"
BS = chr(92)

s = open(R + "sec_sparse_recover.tex", encoding="utf-8").read()
old2 = BS + "section{External Validation (Methodology Note)}\n"
assert old2 in s, "old2 missing"
s = s.replace(old2, "")
open(R + "sec_sparse_recover.tex", "w", encoding="utf-8").write(s)

t = open(R + "sec_scale_deploy.tex", encoding="utf-8").read()
old3 = ("Phase 11A shows unstructured sparsity yields zero memory saving "
        "(5.8" + BS + ",GB flat)" + "\n"
        "and worse energy/token (3.69 $" + BS + "rightarrow$ 6.14" + BS +
        ",J) under dense execution.")
assert old3 in t, "old3 missing"
new3 = ("Unstructured sparsity produced no memory saving or measurable speedup "
        "under dense execution; the observed increase in energy/token is reported "
        "descriptively because generation length and single-run effects remain "
        "confounded.")
t = t.replace(old3, new3)
open(R + "sec_scale_deploy.tex", "w", encoding="utf-8").write(t)

u = open(R + "sec_intro.tex", encoding="utf-8").read()
old4 = "cannot hold 7B+ models in FP16"
assert old4 in u, "old4 missing"
u = u.replace(old4, "cannot generally keep 7B+ FP16 models fully resident in GPU memory")
open(R + "sec_intro.tex", "w", encoding="utf-8").write(u)

f = open(R + "fig_b.tex", encoding="utf-8").read()
i = f.find("% Figure 4")
assert i > 0
f = f[:i].rstrip() + "\n"
open(R + "fig_b.tex", "w", encoding="utf-8").write(f)
print("ALL FIXES APPLIED")
