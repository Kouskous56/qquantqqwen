import re
src = open('D:/qwen/bench/test50_math.py', encoding='utf-8').read()
pat = re.compile(r'\("(\w+)","(\w+)","(.*?)"\s*,\s*\[(.*?)\]\s*,\s*"([ABCD])"\)')
Qs = pat.findall(src)
print('parsed:', len(Qs))
bands = {'b': 'I. Co ban (1-15)', 'h': 'II. Cap 3 (16-30)',
         'o': 'III. Olympic (31-40)', 'a': 'IV. Toan cao cap (41-50)'}
L = ['# De 50 cau trac nghiem toan (co dap an)', '']
cur, n, key = '', 0, []
for cid, band, q, opts, ans in Qs:
    b = bands[cid[0]]
    if b != cur:
        L += [b, '']
        cur = b
    n += 1
    o = re.findall(r'"(.*?)"', opts)
    L.append(f'**Cau {n} ({cid}).** {q}')
    for i, t in enumerate(o):
        L.append(f'- {"ABCD"[i]}. {t}')
    L.append('')
    key.append(f'{n}-{ans}')
L += ['---', '**DAP AN:**', ', '.join(key)]
open('D:/qwen/notes/DE50_MATH.md', 'w', encoding='utf-8').write('\n'.join(L))
print('saved', n)
