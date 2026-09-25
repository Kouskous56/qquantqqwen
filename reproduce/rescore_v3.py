"""Public V3.3 rescore: frozen V3.1 raw generations -> typed scoring ->
v3_cross_scale.csv + McNemar + RUN_V33_RESCORE.json. No GPU needed.
Usage:
  python reproduce/rescore_v3.py --manifest-dir manifests --out-results results/v3_cross_scale.csv --out-manifest manifests/RUN_V33_RESCORE.json
"""
import argparse
import hashlib
import json
import math
import os
import time

from scoring import match

SCALES = ["05B", "15B", "3B"]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-results", required=True)
    ap.add_argument("--out-manifest", required=True)
    a = ap.parse_args()

    Qs = {q["id"]: q for q in json.load(open(a.questions, encoding="utf-8"))}
    with open(a.questions, "rb") as f:
        ds_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    try:
        with open(os.path.join(os.path.dirname(__file__),
                               "scoring.py"), "rb") as f:
            scorer_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        scorer_hash = "unknown"
    src_hashes = {}
    for sc in SCALES:
        for arm in ["S0", "S30"]:
            p = os.path.join(a.manifest_dir, f"RUN_V31_{sc}_{arm}.json")
            src_hashes[f"RUN_V31_{sc}_{arm}.json"] = sha(p)

    rows, table = ["scale,s0_free,s30_free,retention,absolute_drop,n11,n10,n01,n00"], {}
    for sc in SCALES:
        arms = {}
        for arm in ["S0", "S30"]:
            p = os.path.join(a.manifest_dir, f"RUN_V31_{sc}_{arm}.json")
            d = json.load(open(p, encoding="utf-8"))
            per_rep = []
            for rep in d["rep_items"]:
                per_rep.append({it["id"]: match(Qs[it["id"]], it["out"])
                                for it in rep})
            arms[arm] = per_rep
        # stability: all reps identical?
        stab = all(arms[x][0] == arms[x][r] for x in arms for r in (1, 2))
        pa, pb = arms["S0"][0], arms["S30"][0]
        n11 = sum(1 for k in pa if pa[k] and pb[k])
        n10 = sum(1 for k in pa if pa[k] and not pb[k])
        n01 = sum(1 for k in pa if not pa[k] and pb[k])
        n00 = sum(1 for k in pa if not pa[k] and not pb[k])
        sa, sb = sum(pa.values()), sum(pb.values())
        pval = mcnemar_exact(n10, n01)
        table[sc] = {"s0": sa, "s30": sb, "n11": n11, "n10": n10,
                     "n01": n01, "n00": n00, "mcnemar_p": round(pval, 4),
                     "stable_3x": stab}
        rows.append(f"{sc},{sa},{sb},{sb}/{sa},{sa-sb},{n11},{n10},{n01},{n00}")
        print(f"{sc}: {sa}->{sb} p={pval:.4f} stable={stab}", flush=True)
    open(a.out_results, "w", encoding="utf-8").write("\n".join(rows) + "\n")
    res = {"protocol": "V3.3 typed rescore over frozen V3.1 raw generations",
           "note": "V3.1 manifests carry historical pass labels which are NOT "
                   "used here; only frozen raw outputs are rescored. The V3.1 "
                   "generation manifests reference the pre-typed-metadata dataset "
                   "file; question IDs, ordering, and prompt text are unchanged, "
                   "only scoring metadata was added afterwards.",
           "dataset_sha256": ds_hash, "scorer": "reproduce/scoring.py",
           "scorer_sha256": scorer_hash,
           "source_manifests": src_hashes,
           "scores": table, "timestamp": time.strftime("%Y-%m-%dT%H:%M")}
    json.dump(res, open(a.out_manifest, "w", encoding="utf-8"), indent=1)
    print(f"wrote {a.out_results} + {a.out_manifest}")


main()
