import re
p = "D:/qwen/report/paper.tex"
t = open(p, encoding="utf-8").read()
old = ("\\caption{Deployment VRAM (left) and B-rate under the legacy generative "
       "V2 protocol, exploratory \nonly (right).}")
assert old in t, "caption not found"
t = t.replace(old, "\\caption{Deployment VRAM by model (Ollama, Q4).}")
open(p, "w", encoding="utf-8").write(t)
print("caption fixed")
