import argparse
import json
from pathlib import Path
import pytest
from reproduce.scoring_v401 import score, symbolic
from reproduce.eval_gsm import extract, extract_numeric
from bench.qwen3 import common


@pytest.mark.parametrize('value', ['x^23', 'x^{23}', '3x^23', '3*x^2*', '23', '2*3'])
def test_derivative_rejects_false_alias_collisions(value):
    assert not score({'answer_type': 'symbolic_alias', 'accepted': ['3x^2', 'x^2*3']}, value)['pass']


@pytest.mark.parametrize('value', ['3x^2', '3*x^2', r'3\cdot x^{2}', 'x^2*3', r'x^{2}\times3'])
def test_derivative_accepts_declared_equivalents(value):
    assert score({'answer_type': 'symbolic_alias', 'accepted': ['3x^2', 'x^2*3']}, value)['pass']


def test_numeric_boundaries_remain_distinct():
    assert symbolic('2*3') != symbolic('23')
    assert symbolic('x^2*3') != symbolic('x^23')


@pytest.mark.parametrize('text,expected', [
    (r'First \boxed{2}. Correction: #### 3', '3'),
    (r'#### 2 Correction: \boxed{3}', '3'),
    (r'\boxed{2} #### invalid', ''), (r'#### 2 \boxed{invalid}', ''),
    (r'#### 2 \boxed{3', ''), ('#### 3.4.5', ''), ('3.4.5', ''),
    ('#### 3..4', ''), ('#### 1,23', ''), ('#### 1/2/3', ''),
    ('#### 3.4. Explanation.', '17/5'), ('#### 3.', '3'),
    ('#### 1e99999999', ''), ('#### 1e-2', '1/100'),
])
def test_gsm_last_marker_and_malformed_tokens(text, expected):
    assert extract_numeric(text) == expected


def test_historical_gsm_semantics_unchanged():
    assert extract(r'\boxed{2} Correction: #### 3') == '2'


@pytest.fixture
def served(tmp_path, monkeypatch):
    blob = tmp_path / 'model.gguf'
    blob.write_bytes(b'fixture-weights')
    digest = common.sha256(blob)
    args = argparse.Namespace(model='same-alias', endpoint='http://localhost:11434', tag='test', gguf=str(blob))
    state = {'manifest': 'a'*64, 'template': 'template-v1', 'gguf': digest, 'calls': 0}
    def api(endpoint, route, payload=None):
        if route == 'tags':
            return {'models': [{'name': 'same-alias:latest', 'digest': state['manifest']}]}
        if route == 'show':
            return {'modelfile': 'FROM /blobs/sha256-' + state['gguf'], 'template': state['template'], 'parameters': 'temperature 0'}
        return {'version': 'test-version'}
    monkeypatch.setattr(common, 'model_api', api)
    return args, state, blob


def test_artifact_bytes_must_match_served_blob(served):
    args, state, blob = served
    assert common.model_binding(args)['gguf_sha256'] == state['gguf']
    blob.write_bytes(b'different weights')
    with pytest.raises(ValueError, match='GGUF bytes'):
        common.model_binding(args)


@pytest.mark.parametrize('field', ['manifest', 'template', 'gguf'])
def test_resume_rejects_same_alias_with_changed_artifact_or_template(served, tmp_path, monkeypatch, field):
    args, state, blob = served
    items = [(i, {'question': str(i), 'answer': '#### 2'}) for i in range(2)]
    checkpoint = tmp_path/'checkpoint.json'
    def chat(*args):
        state['calls'] += 1
        if state['calls'] == 2:
            raise RuntimeError('interrupted')
        return '#### 2'
    monkeypatch.setattr(common, 'chat', chat)
    with pytest.raises(RuntimeError):
        common.evaluate(args, items, 'gsm', checkpoint, __file__, 32)
    original = checkpoint.read_bytes()
    state[field] = 'b'*64
    with pytest.raises(ValueError):
        common.evaluate(args, items, 'gsm', checkpoint, __file__, 32)
    assert checkpoint.read_bytes() == original
    assert state['calls'] == 2


def test_alias_swap_during_response_not_saved(served, tmp_path, monkeypatch):
    args, state, _ = served
    def chat(*args):
        state['manifest'] = 'b'*64
        return '#### 2'
    monkeypatch.setattr(common, 'chat', chat)
    checkpoint = tmp_path/'checkpoint.json'
    with pytest.raises(ValueError, match='during generation'):
        common.evaluate(args, [(0, {'question': '1+1', 'answer': '#### 2'})], 'gsm', checkpoint, __file__, 32)
    assert not checkpoint.exists()


def test_legacy_checkpoint_without_artifact_identity_is_rejected(served, tmp_path):
    args, _, _ = served
    items = [(0, {'question': '1+1', 'answer': '#### 2'})]
    identity = common.run_identity(args, items, 'gsm', __file__)
    old = dict(identity)
    old.pop('model_binding')
    old['version'] = 1
    checkpoint = tmp_path/'checkpoint.json'
    common.atomic_json(checkpoint, {'identity': old, 'done': 0, 'details': [], 'elapsed_s': 0})
    with pytest.raises(ValueError, match='different or legacy'):
        common.load_checkpoint(checkpoint, identity, items, 'gsm')
