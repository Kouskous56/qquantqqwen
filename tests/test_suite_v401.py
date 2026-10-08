import copy
import json
from collections import Counter
from pathlib import Path
import pytest
from reproduce.scoring_v401 import score, number
from reproduce.suite_v401 import ROOT, load, plan, digest, summarize
from reproduce.eval_suite_v401 import run, validate_records, Ollama
from reproduce.rescore_suite_v401 import audit, markdown


def question(kind='numeric_scalar', accepted=None):
    return {'answer_type': kind, 'accepted': accepted or ['4']}


@pytest.mark.parametrize('raw', ['4', '+4', '4.0', '8/2', '4e0', r'\frac{8}{2}',
                               r'\boxed{4}', '$4$', '4<|im_end|>', 'Answer: 4'])
def test_exact_numeric_variants(raw):
    assert score(question(), raw)['pass']


@pytest.mark.parametrize('raw', ['The answer is 4', '2+2=4', '4 kg', '4x', '4/0', 'NaN', 'inf',
                               '4 then 5', '1e999', '4,00', '4<|im_start|>', r'\boxed{4} text'])
def test_reject_extraction_and_invalid_numbers(raw):
    assert not score(question(), raw)['pass']


def test_exact_not_tolerance_and_format_independent_of_correctness():
    assert not score(question(), '4.000000001')['pass']
    assert score(question(), '5')['format_compliant']
    assert score(question(), 'Answer: 4')['pass']
    assert not score(question(), 'Answer: 4')['format_compliant']
    assert score(question(), '')['status'] == 'empty'
    assert number('1,000') == '1000'


def test_multiset_multiplicity_and_tuple_order():
    q = question('unordered_numeric_multiset', ['1,1'])
    assert score(q, '(1,1)')['pass']
    assert not score(q, '1')['pass']
    assert not score(q, '1,1,1')['pass']
    q = question('unordered_numeric_multiset', ['4,6'])
    assert score(q, '6,4')['pass']
    q = question('ordered_numeric_tuple', ['3,2'])
    assert score(q, '(3,2)')['pass']
    assert not score(q, '(2,3)')['pass']
    assert not score(q, '(3,2]')['pass']


def test_symbolic_policy():
    q = question('symbolic_alias', ['3x^2'])
    assert score(q, r'3\cdot x^{2}')['pass']
    assert not score(q, '3***x^2')['pass']
    assert not score(q, '3x^2*')['pass']
    assert not score(q, '*3x^2')['pass']
    assert not score(q, 'The answer is 3x^2')['pass']
    assert not score(q, 'x^2+x^2+x^2')['pass']  # documented finite aliases


def test_mcq_strict_and_invalid_gold():
    assert score({}, 'b', 'mcq', 'B')['pass']
    assert not score({}, 'B. 4', 'mcq', 'B')['pass']
    for bad in (None, '', 'AB', 1, 'E'):
        with pytest.raises(ValueError):
            score({}, 'A', 'mcq', bad)


def test_frozen_suite_and_all_choices():
    suite, _ = load()
    assert len(suite['questions']) == 50
    assert sum(not q['review']['legacy_question_unchanged'] for q in suite['questions']) == 18
    for q in suite['questions']:
        assert all(score(q, a)['pass'] for a in q['accepted'])
        assert [score(q, c)['pass'] for c in q['choices']] == [letter == q['answer'] for letter in 'ABCD']


def test_rotation_plan_balanced_and_reproducible():
    free, mcq = plan('free'), plan('mcq')
    assert len(free['items']) == 50 and len(mcq['items']) == 200
    assert Counter(i['gold'] for i in mcq['items']) == dict.fromkeys('ABCD', 50)
    assert mcq == plan('mcq')
    modified = copy.deepcopy(mcq['items'])
    modified[0]['prompt'] += ' changed'
    assert digest(modified) != mcq['prompt_plan_sha256']
    for q in load()[0]['questions']:
        for item in [i for i in mcq['items'] if i['id'] == q['id']]:
            assert f"{item['gold']}. {q['choices']['ABCD'.index(q['answer'])]}" in item['prompt']


class FakeBackend:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at
    def identity(self):
        return {'backend': 'TEST-ONLY', 'model': 'fixture'}
    def generate(self, item, generation):
        if self.calls == self.fail_at:
            raise RuntimeError('simulated interruption')
        self.calls += 1
        q = next(q for q in load()[0]['questions'] if q['id'] == item['id'])
        return {'raw': item['gold'] or q['accepted'][0], 'finish_reason': 'stop',
                'output_tokens': 1, 'prompt_tokens': 20}


@pytest.mark.parametrize('mode,total', [('free', 50), ('mcq', 200)])
def test_runner_atomic_checkpoint_resume_and_summary(tmp_path, mode, total):
    output = tmp_path / 'run.json'
    with pytest.raises(RuntimeError):
        run(FakeBackend(fail_at=3), mode, output)
    partial = json.loads(output.read_text())
    assert len(partial['records']) == 3 and not partial['complete']
    backend = FakeBackend()
    final = run(backend, mode, output, resume=True)
    assert backend.calls == total - 3
    assert final['complete'] and final['summary']['correct'] == total
    if mode == 'mcq':
        assert final['summary']['all_four_correct'] == 50
    with pytest.raises(ValueError, match='already exists'):
        run(backend, mode, output)
    final['identity']['model']['model'] = 'different'
    output.write_text(json.dumps(final))
    with pytest.raises(ValueError, match='identity mismatch'):
        run(backend, mode, output, resume=True)


def test_corrupt_checkpoint_rejected(tmp_path):
    output = tmp_path / 'run.json'
    result = run(FakeBackend(), 'free', output)
    for key, value in [('prompt_sha256', 'bad'), ('raw', 'wrong'), ('output_tokens', -1)]:
        corrupt = copy.deepcopy(result)
        corrupt['records'][0][key] = value
        output.write_text(json.dumps(corrupt))
        with pytest.raises(ValueError):
            run(FakeBackend(), 'free', output, True)


def test_model_changes_rejected_before_checkpoint(tmp_path):
    class Changing(FakeBackend):
        def identity(self):
            return {'model': self.calls}
    with pytest.raises(ValueError, match='changed'):
        run(Changing(), 'free', tmp_path / 'run.json')
    assert not (tmp_path / 'run.json').exists()


def test_lock_detects_tampering(tmp_path, monkeypatch):
    import reproduce.suite_v401 as module
    directory = tmp_path / 'data/v4_0_1'
    directory.mkdir(parents=True)
    (directory / 'release.lock.json').write_text(json.dumps({'files': {'tampered': 'wrong'}}))
    (tmp_path / 'tampered').write_text('content')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Frozen release modified'):
        module.load()


def test_ollama_requires_localhost_and_gguf_binding(tmp_path):
    with pytest.raises(ValueError, match='local HTTP'):
        Ollama('http://example.com', 'model', tmp_path / 'missing')
    gguf = tmp_path / 'fixture.gguf'
    gguf.write_bytes(b'test-only')
    backend = Ollama('http://localhost:11434', 'model', gguf)
    backend.request = lambda endpoint, data=None: {'models': [{'name': 'model:latest', 'digest': 'manifest'}]} if endpoint == 'tags' else {'modelfile': 'FROM sha256-' + '0' * 64}
    with pytest.raises(ValueError, match='GGUF SHA256'):
        backend.identity()


def test_ollama_identity_and_request_contract(tmp_path):
    gguf = tmp_path / 'fixture.gguf'
    gguf.write_bytes(b'test-only')
    backend = Ollama('http://localhost:11434', 'model', gguf)
    calls = []
    def request(endpoint, data=None):
        calls.append((endpoint, data))
        return {'tags': {'models': [{'name': 'model:latest', 'digest': 'manifest-hash'}]},
                'show': {'modelfile': 'FROM "C:/blobs/sha256-' + backend.gguf_sha256 + '"', 'template': 'template'},
                'version': {'version': 'test'},
                'chat': {'done': True, 'done_reason': 'stop', 'message': {'content': '4'},
                         'eval_count': 1, 'prompt_eval_count': 20}}[endpoint]
    backend.request = request
    identity = backend.identity()
    assert identity['manifest_sha256'] != identity['gguf_sha256']
    p = plan('free')
    assert backend.generate(p['items'][0], p['generation'])['raw'] == '4'
    payload = calls[-1][1]
    assert payload['stream'] is False and payload['think'] is False
    assert payload['options']['num_predict'] == 64
    assert payload['options']['temperature'] == 0
    assert len(payload['messages']) == 1


@pytest.mark.parametrize('patch', [
    {'done': False}, {'done_reason': None}, {'eval_count': 65},
    {'prompt_eval_count': 2048}, {'eval_count': -1},
    {'message': {'content': '4', 'thinking': 'hidden reasoning'}},
])
def test_ollama_rejects_protocol_violations(tmp_path, patch):
    gguf = tmp_path / 'fixture.gguf'
    gguf.write_bytes(b'test-only')
    backend = Ollama('http://localhost:11434', 'model', gguf)
    response = {'done': True, 'done_reason': 'stop', 'message': {'content': '4'},
                'eval_count': 1, 'prompt_eval_count': 20}
    response.update(patch)
    backend.request = lambda *args: response
    p = plan('free')
    with pytest.raises(ValueError):
        backend.generate(p['items'][0], p['generation'])


def test_migration_is_deterministic_and_not_new_experiment():
    result = audit()
    assert result['is_v4_experiment'] is False
    assert len(result['rows']) == 24
    assert len(result['excluded_changed_stems']) == 18
    assert all(r['unchanged_stems_count'] == 32 for r in result['rows'])
    assert result == audit()
    assert not any(d['id'] == 'a50' and d['v3_recomputed'] != d['v4_rules_old_key']['pass'] for d in result['changed_details'])
    frozen = ROOT / 'results/v4_0_1_migration/migration.json'
    if frozen.exists():
        assert result == json.loads(frozen.read_text(encoding='utf-8'))
        assert markdown(result) == frozen.with_name('MIGRATION.md').read_text(encoding='utf-8')
