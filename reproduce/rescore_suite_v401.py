"""Deterministic migration audit; never presents historical outputs as V4 runs."""
import argparse
import json
from pathlib import Path
from reproduce.scoring import match
from reproduce.scoring_v401 import score
from reproduce.suite_v401 import ROOT, file_hash, load


def audit():
    suite, _ = load()
    current = {q['id']: q for q in suite['questions']}
    legacy = {q['id']: q for q in json.loads((ROOT / 'data/v2/questions.json').read_text(encoding='utf-8'))}
    files = sorted((ROOT / 'manifests').glob('RUN_V31_*_S*.json'))
    files += sorted((ROOT / 'manifests_qwen3').glob('RUN_QWEN3_TYPED_*.json'))
    rows, details, hashes = [], [], {}
    for path in files:
        data = json.loads(path.read_text(encoding='utf-8'))
        hashes[path.relative_to(ROOT).as_posix()] = file_hash(path)
        repetitions = data.get('rep_items', [data.get('details')])
        for rep, items in enumerate(repetitions, 1):
            if not isinstance(items, list) or len(items) != 50 or {i['id'] for i in items} != set(legacy):
                raise ValueError('Invalid historical item coverage: ' + str(path))
            row = {'source': path.name, 'rep': rep, 'historical_labels': 0,
                   'v3_recomputed': 0, 'v4_rules_old_keys': 0,
                   'unchanged_stems_count': 0, 'unchanged_stems_new_keys_correct': 0,
                   'gained': 0, 'lost': 0}
            for item in items:
                qid = item['id']
                raw = item.get('out', item.get('raw'))
                old = legacy[qid]
                mapped = dict(old)
                mapped['answer_type'] = {'unordered_numeric_set': 'unordered_numeric_multiset',
                                        'symbolic_exact': 'symbolic_alias'}.get(old['answer_type'], old['answer_type'])
                if current[qid]['answer_type'] == 'text_label':
                    mapped['answer_type'] = 'text_label'
                before = match(old, raw)
                after = score(mapped, raw)
                row['historical_labels'] += bool(item['pass'])
                row['v3_recomputed'] += before
                row['v4_rules_old_keys'] += after['pass']
                row['gained'] += after['pass'] and not before
                row['lost'] += before and not after['pass']
                eligible = current[qid]['review']['legacy_question_unchanged']
                if eligible:
                    row['unchanged_stems_count'] += 1
                    row['unchanged_stems_new_keys_correct'] += score(current[qid], raw)['pass']
                if before != after['pass'] or before != bool(item['pass']):
                    details.append({'source': path.name, 'rep': rep, 'id': qid, 'raw': raw,
                        'historical_label': item['pass'], 'v3_recomputed': before,
                        'v4_rules_old_key': after,
                        'reason': 'Whole-response parsing / exact numeric or finite-alias policy differs from V3 extraction'
                                  if before != after['pass'] else 'Historical stored label differs from canonical V3 scorer'})
            rows.append(row)
    return {'kind': 'historical-scorer-migration-only', 'is_v4_experiment': False,
            'input_sha256': hashes,
            'legacy_dataset_sha256': file_hash(ROOT / 'data/v2/questions.json'),
            'legacy_scorer_sha256': file_hash(ROOT / 'reproduce/scoring.py'),
            'release': json.loads((ROOT / 'data/v4_0_1/release.lock.json').read_text(encoding='utf-8')),
            'excluded_changed_stems': [q['id'] for q in current.values() if not q['review']['legacy_question_unchanged']],
            'caveat': 'Even unchanged stems used historical instructions, budgets and backends. None of these scores is a V4 experiment. Qwen3 original scorer provenance remains unknown.',
            'rows': rows, 'changed_details': details}


def markdown(report):
    lines = ['# Math50 V4 migration audit', '', report['caveat'], '',
             'Stage A recomputes canonical V3 scoring. Stage B applies V4 rules to OLD keys and OLD outputs; no question edits are mixed into this difference. Stage C audits only the 32 unchanged stems with new aliases; it is still not a V4 run.', '',
             '| Source | Rep | Stored | V3 A /50 | V4 rules B /50 | Gained | Lost | C /32 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['rows']:
        lines.append(f"| {r['source']} | {r['rep']} | {r['historical_labels']} | {r['v3_recomputed']} | {r['v4_rules_old_keys']} | {r['gained']} | {r['lost']} | {r['unchanged_stems_new_keys_correct']} |")
    lines += ['', 'Changed stems excluded from stage C: ' + ', '.join(report['excluded_changed_stems']), '',
              'The companion JSON contains full source hashes and every changed judgment with its original public raw output. Repeated deterministic generations are not independent samples. Score losses here establish a parsing-policy difference, not a model-quality regression.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.out_dir / 'migration.json', args.out_dir / 'MIGRATION.md']
    if any(p.exists() for p in paths):
        parser.error('Use a new output directory; do not overwrite historical reports')
    report = audit()
    paths[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    paths[1].write_text(markdown(report), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
