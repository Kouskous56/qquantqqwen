import json
from collections import Counter
R = json.load(open('D:/qwen/notes/TEST50_V2.json', encoding='utf-8'))
for n in ['F16', 'Q4', 'qwen3b-mine-q4', 'qwen2.5:3b']:
    d = R[n]
    # rotation-wise accuracy
    rot = {}
    for r in range(4):
        ok = sum(1 for cid in d['raw'] if d['raw'][cid].get(f'r{r}', {}).get('got') == d['raw'][cid].get(f'r{r}', {}).get('exp'))
        tot = sum(1 for cid in d['raw'] if f'r{r}' in d['raw'][cid])
        rot[r] = f'{ok}/{tot}'
    # k/4 distribution
    kd = Counter()
    for cid in d['raw']:
        k = sum(1 for r in range(4) if d['raw'][cid].get(f'r{r}', {}).get('got') == d['raw'][cid].get(f'r{r}', {}).get('exp'))
        kd[k] += 1
    # format: single char output?
    outs = [d['raw'][cid][f'r{r}']['out'] for cid in d['raw'] for r in range(4) if f'r{r}' in d['raw'][cid]]
    single = sum(1 for o in outs if len(o.strip()) == 1 and o.strip() in 'ABCD')
    youare = sum(1 for o in outs if 'You are an' in o)
    ansb = sum(1 for o in outs if 'Answer: B' in o)
    print(f'== {n} == rot:{rot} kdist:{dict(sorted(kd.items()))} single:{single}/{len(outs)} youare:{youare} ansB:{ansb}')
# mine vs official
a, b = R['qwen3b-mine-q4']['raw'], R['qwen2.5:3b']['raw']
same_l = sum(1 for cid in a for r in range(4) if a[cid][f'r{r}']['got'] == b[cid][f'r{r}']['got'])
same_c = sum(1 for cid in a for r in range(4) if (a[cid][f'r{r}']['got'] == a[cid][f'r{r}']['exp']) == (b[cid][f'r{r}']['got'] == b[cid][f'r{r}']['exp']))
mw = sum(1 for cid in a for r in range(4) if (a[cid][f'r{r}']['got'] == a[cid][f'r{r}']['exp']) and not (b[cid][f'r{r}']['got'] == b[cid][f'r{r}']['exp']))
ow = sum(1 for cid in a for r in range(4) if (b[cid][f'r{r}']['got'] == b[cid][f'r{r}']['exp']) and not (a[cid][f'r{r}']['got'] == a[cid][f'r{r}']['exp']))
print('same-letter:', same_l, '/200 same-correctness:', same_c, '/200 mine-win:', mw, 'off-win:', ow)
# consistent sets
def cons(x):
    return {cid for cid in x if all(x[cid][f'r{r}']['got'] == x[cid][f'r{r}']['exp'] for r in range(4))}
ca, cb = cons(a), cons(b)
print('consistent mine:', len(ca), 'off:', len(cb), 'shared:', len(ca & cb), 'mine-only:', ca - cb, 'off-only:', cb - ca)
# o36
for m in ['qwen3b-mine-q4', 'qwen2.5:3b']:
    print(m, 'o36:', {f"r{r}": (R[m]['raw']['o36'][f'r{r}']['got'], R[m]['raw']['o36'][f'r{r}']['exp']) for r in range(4)}, 'free:', R[m]['raw']['o36']['free'][:60])
# F16/Q4 agreement
fa, qa = R['F16']['raw'], R['Q4']['raw']
print('F16/Q4 same-letter:', sum(1 for cid in fa for r in range(4) if fa[cid][f'r{r}']['got'] == qa[cid][f'r{r}']['got']), '/200')
# sections (basic b, hs h, olymp o, higher a) over 200 perm samples
for m in ['F16', 'Q4', 'qwen3b-mine-q4', 'qwen2.5:3b']:
    d = R[m]['raw']
    for band, pre in [('basic', 'b'), ('hs', 'h'), ('olymp', 'o'), ('higher', 'a')]:
        ok = sum(1 for cid in d if cid.startswith(pre) for r in range(4) if d[cid][f'r{r}']['got'] == d[cid][f'r{r}']['exp'])
        tot = sum(1 for cid in d if cid.startswith(pre)) * 4
        print(m, band, f'{ok}/{tot}={ok/tot:.1%}')
