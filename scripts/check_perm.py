import json, os
d1 = "D:/qwen/notes/lmeval/mb05dense/D__qwen__models__Qwen2.5-0.5B-FP16"
d2 = "D:/qwen/notes/lmeval/mb05dense_perm/D__qwen__models__Qwen2.5-0.5B-FP16"
f1 = [f for f in os.listdir(d1) if "arc_" in f][0]
f2 = [f for f in os.listdir(d2) if "arc_" in f][0]
ja = json.loads(open(os.path.join(d1, f1), encoding="utf-8").readline())
jb = json.loads(open(os.path.join(d2, f2), encoding="utf-8").readline())
print("ORIG:", {k: str(v)[:80] for k, v in ja["doc"].items() if k != "choices"})
print("PERM:", {k: str(v)[:80] for k, v in jb["doc"].items() if k != "choices"})
print("orig choices:", ja["doc"].get("choices"))
print("perm choices:", jb["doc"].get("choices"))
print("target orig:", ja.get("target"), "| perm:", jb.get("target"))
