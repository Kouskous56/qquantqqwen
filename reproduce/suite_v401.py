"""Frozen Math50 protocol, prompt plans and content identities (stdlib only)."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from reproduce.scoring_v401 import VERSION, TYPES, normalize, score

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(verify=True):
    if verify:
        lock = json.loads((ROOT / 'data/v4_0_1/release.lock.json').read_text(encoding='utf-8'))
        for name, expected in lock['files'].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError('Frozen release modified: ' + name)
    suite = json.loads((ROOT / 'data/v4_0_1/questions.json').read_text(encoding='utf-8'))
    protocol = json.loads((ROOT / 'data/v4_0_1/protocol.json').read_text(encoding='utf-8'))
    questions = suite['questions']
    if len(questions) != 50 or len({q['id'] for q in questions}) != 50:
        raise ValueError('Require 50 distinct questions')
    if protocol['suite_id'] != suite['suite_id'] or protocol['scorer_version'] != VERSION:
        raise ValueError('Protocol identity mismatch')
    for q in questions:
        if q['answer_type'] not in TYPES or not q['accepted'] or len(q['choices']) != 4:
            raise ValueError('Invalid question: ' + q['id'])
        if q['answer'] not in list('ABCD'):
            raise ValueError('Invalid answer key')
        if any(normalize(q['answer_type'], a) in (None, '') for a in q['accepted']):
            raise ValueError('Invalid accepted answer')
        matches = [score(q, choice)['pass'] for choice in q['choices']]
        if sum(matches) != 1 or not matches['ABCD'.index(q['answer'])]:
            raise ValueError('Choices must have exactly one correct answer: ' + q['id'])
    return suite, protocol


def plan(mode, verify=True):
    suite, protocol = load(verify)
    if mode not in protocol['modes']:
        raise ValueError('Unknown mode')
    items = []
    for q in suite['questions']:
        for rotation in protocol['modes'][mode].get('rotations', [0]):
            prompt = q['question']
            gold = None
            if mode == 'mcq':
                choices = q['choices'][rotation:] + q['choices'][:rotation]
                prompt += '\n' + '\n'.join(f'{letter}. {choice}' for letter, choice in zip('ABCD', choices))
                prompt += '\n' + protocol['mcq_instruction']
                gold = 'ABCD'[('ABCD'.index(q['answer']) - rotation) % 4]
            else:
                prompt += '\n' + protocol['free_instructions'][q['answer_type']]
            items.append({'id': q['id'], 'rotation': rotation, 'prompt': prompt,
                          'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'gold': gold})
    return {'suite_id': suite['suite_id'], 'protocol_id': protocol['protocol_id'],
            'mode': mode, 'items': items, 'prompt_plan_sha256': digest(items),
            'generation': {**protocol['generation'], **protocol['modes'][mode]}}


def summarize(records, mode):
    status = Counter(r['score']['status'] for r in records)
    result = {'responses': len(records), 'correct': status['correct'], 'statuses': dict(status),
              'termination': dict(Counter(r['finish_reason'] for r in records))}
    result['by_band'] = {band: {'responses': sum(r['band'] == band for r in records),
        'correct': sum(r['band'] == band and r['score']['pass'] for r in records)}
        for band in sorted({r['band'] for r in records})}
    if mode == 'mcq':
        grouped = {}
        for r in records:
            grouped.setdefault(r['id'], []).append(r['score']['pass'])
        result['all_four_correct'] = sum(len(v) == 4 and all(v) for v in grouped.values())
        result['answer_letters'] = dict(Counter(r['score']['normalized'] or 'invalid' for r in records))
    return result
