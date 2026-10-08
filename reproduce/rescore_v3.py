"""Public V3.3 rescore: frozen V3.1 raw generations -> typed scoring ->
v3_cross_scale.csv + McNemar + RUN_V33_RESCORE.json. No GPU needed.
Usage:
  python reproduce/rescore_v3.py --manifest-dir manifests --questions data/v2/questions.json --out-results results/v3_cross_scale.csv --out-manifest manifests/RUN_V33_RESCORE.json
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

if __package__:
    from .evaluation_io import validate_questions, write_json
    from .scoring import match
else:
    from evaluation_io import validate_questions, write_json
    from scoring import match

SCALES = ["05B", "15B", "3B"]


def mcnemar_exact(b, c):
    if any(type(x) is not int or x < 0 for x in (b, c)):
        raise ValueError("discordant counts must be non-negative integers")
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


def score_repetitions(data, questions, source):
    """Validate pairing before scoring; never silently drop or duplicate items."""
    reps = data.get("rep_items")
    if not isinstance(reps, list) or len(reps) != 3:
        raise ValueError(f"{source}: canonical V3.3 requires exactly three repetitions")
    scored = []
    expected = set(questions)
    for index, rep in enumerate(reps, start=1):
        if not isinstance(rep, list):
            raise ValueError(f"{source}: repetition {index} must be a list")
        values = {}
        for item in rep:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError(f"{source}: repetition {index} has an invalid item id")
            qid = item["id"]
            if qid in values:
                raise ValueError(f"{source}: repetition {index} has duplicate id {qid}")
            if qid not in questions:
                raise ValueError(f"{source}: repetition {index} has unknown id {qid}")
            if not isinstance(item.get("out"), str):
                raise ValueError(f"{source}: repetition {index}, {qid}: out must be text")
            values[qid] = match(questions[qid], item["out"])
        missing = expected - values.keys()
        if missing:
            raise ValueError(f"{source}: repetition {index} is missing IDs: {sorted(missing)}")
        scored.append(values)
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-results", required=True)
    ap.add_argument("--out-manifest", required=True)
    a = ap.parse_args()

    qbytes = Path(a.questions).read_bytes()
    Qs = {q["id"]: q for q in validate_questions(json.loads(qbytes), typed=True)}
    ds_hash = hashlib.sha256(qbytes).hexdigest()[:12]
    scorer_hash = sha(Path(__file__).with_name("scoring.py"))
    src_hashes = {}
    sources = {}
    for sc in SCALES:
        for arm in ["S0", "S30"]:
            p = os.path.join(a.manifest_dir, f"RUN_V31_{sc}_{arm}.json")
            raw = Path(p).read_bytes()
            src_hashes[f"RUN_V31_{sc}_{arm}.json"] = hashlib.sha256(raw).hexdigest()[:12]
            sources[(sc, arm)] = score_repetitions(json.loads(raw), Qs, p)

    rows, table = ["scale,s0_free,s30_free,retention,absolute_drop,n11,n10,n01,n00"], {}
    for sc in SCALES:
        arms = {}
        for arm in ["S0", "S30"]:
            arms[arm] = sources[(sc, arm)]
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
    out_results = Path(a.out_results)
    out_results.parent.mkdir(parents=True, exist_ok=True)
    with out_results.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
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
    write_json(a.out_manifest, res)
    print(f"wrote {a.out_results} + {a.out_manifest}")


if __name__ == "__main__":
    main()
