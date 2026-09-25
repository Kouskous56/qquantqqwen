t = open("D:/qwen/report/paper.tex", encoding="utf-8").read()
fig2 = ("\n\\begin{figure*}[t]\n\\centering\n\\input{fig_b}\n"
        "\\caption{Deployment VRAM (left) and positional bias (right).}\n"
        "\\label{fig:deploy}\n\\end{figure*}\n")
assert "\\bibliographystyle{plain}" in t
t = t.replace("\\bibliographystyle{plain}", fig2 + "\\bibliographystyle{plain}")
open("D:/qwen/report/paper.tex", "w", encoding="utf-8").write(t)
print("inserted fig_b OK")
