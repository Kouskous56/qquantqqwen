import json, re
from collections import Counter
d = json.load(open('D:/qwen/notes/TEST50_MATH.json'))
# lay dap an chuan tu script goc
src = open('D:/qwen/bench/test50_math.py', encoding='utf-8').read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = pat.findall(src)
exp = {cid: ans for cid, _, _, _, ans in Qs}
got = {}
for m, v in d.items():
    g = {}
    wrong = {}
    for w in v['wrong']:
        mm = re.match(r'(\w+)\(got ([ABCD?]), exp ([ABCD])\)', w)
        if mm: wrong[mm.group(1)] = mm.group(2)
    for cid, _, _, _, ans in Qs:
        g[cid] = wrong.get(cid, exp[cid])
    got[m] = g
print('== B-rate ==')
for m, g in got.items():
    c = Counter(g.values())
    print(m, dict(c), 'B-rate', round(c['B']/50, 3))
print('== agreement vs F16 (0.5B) ==')
for m in ['Q8', 'Q4', 'Q3']:
    ag = sum(1 for cid in exp if got[m][cid] == got['F16'][cid])
    print(m, ag, '/50')
a = sum(1 for cid in exp if got['qwen3b-mine-q4'][cid] == got['qwen2.5:3b'][cid])
print('mine-q4 vs official agreement:', a, '/50')
print('== option-conditional acc ==')
for m, g in got.items():
    per = {}
    for opt in 'ABCD':
        ids = [cid for cid in exp if exp[cid] == opt]
        per[opt] = f'{sum(1 for cid in ids if g[cid]==opt)}/{len(ids)}'
    print(m, per)
