"""Run frozen Math50 V4 against a local Ollama model bound to a GGUF hash.

Usage: python -m reproduce.eval_suite_v401 --mode free --prepare-only --out prompts.json
       python -m reproduce.eval_suite_v401 --mode free --model NAME --gguf MODEL.gguf --out run.json
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import urllib.request
from urllib.parse import urlparse
from reproduce.scoring_v401 import score
from reproduce.suite_v401 import ROOT, digest, file_hash, load, plan, summarize


def write_atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Ollama:
    def __init__(self, url, model, gguf):
        parsed = urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('Use a local HTTP Ollama server')
        self.url, self.model = url.rstrip('/'), model
        self.gguf = Path(gguf).resolve(strict=True)
        self.gguf_sha256 = file_hash(self.gguf)

    def request(self, endpoint, data=None):
        request = urllib.request.Request(self.url + '/api/' + endpoint,
                  data=None if data is None else json.dumps(data).encode(),
                  headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def identity(self):
        name = self.model if ':' in self.model else self.model + ':latest'
        entries = [m for m in self.request('tags')['models'] if m['name'] == name]
        if len(entries) != 1:
            raise ValueError('Require one exact local model tag: ' + name)
        info = self.request('show', {'model': self.model})
        # A model manifest digest is NOT a GGUF digest. Bind the FROM blob too.
        sources = re.findall(r'^FROM\s+(.+)$', info.get('modelfile', ''), re.M)
        if len(sources) != 1 or not re.search(r'sha256[-:]' + self.gguf_sha256 + r'(?:["\s]|$)', sources[0]):
            raise ValueError('Ollama FROM blob does not match supplied GGUF SHA256')
        if re.search(r'^ADAPTER\s', info.get('modelfile', ''), re.M):
            raise ValueError('Adapter layers require a separately versioned backend policy')
        return {'backend': 'ollama-chat-v1', 'version': self.request('version')['version'],
                'model_tag': name, 'manifest_sha256': entries[0]['digest'],
                'gguf_sha256': self.gguf_sha256, 'show_sha256': digest(info),
                'template_sha256': digest(info.get('template', '')),
                'parameters': info.get('parameters', ''), 'system': info.get('system', ''),
                'host': platform.platform(), 'python': platform.python_version()}

    def generate(self, item, generation):
        response = self.request('chat', {'model': self.model, 'stream': False, 'think': False,
            'messages': [{'role': 'user', 'content': item['prompt']}],
            'options': {'seed': generation['seed'], 'temperature': 0, 'top_k': 1,
                        'top_p': 1, 'repeat_penalty': 1.0, 'num_ctx': generation['num_ctx'],
                        'num_predict': generation['max_new_tokens']}})
        if response.get('done') is not True or not isinstance(response.get('message', {}).get('content'), str):
            raise ValueError('Incomplete backend response')
        for key in ('eval_count', 'prompt_eval_count'):
            if type(response.get(key)) is not int or response[key] < 0:
                raise ValueError('Backend must report token counts')
        if response.get('done_reason') not in {'stop', 'length'}:
            raise ValueError('Backend must report stop or length termination')
        if response['eval_count'] > generation['max_new_tokens']:
            raise ValueError('Backend exceeded generation budget')
        if response['prompt_eval_count'] + generation['max_new_tokens'] > generation['num_ctx']:
            raise ValueError('Prompt plus generation budget exceeds context')
        if response['message'].get('thinking'):
            raise ValueError('Thinking output violates frozen non-thinking policy')
        return {'raw': response['message']['content'], 'finish_reason': response['done_reason'],
                'output_tokens': response['eval_count'], 'prompt_tokens': response['prompt_eval_count']}


def validate_records(records, prompt_plan, questions):
    if not isinstance(records, list) or len(records) > len(prompt_plan['items']):
        raise ValueError('Invalid checkpoint length')
    for saved, item in zip(records, prompt_plan['items']):
        if any(saved.get(key) != item[key] for key in ('id', 'rotation', 'prompt_sha256')):
            raise ValueError('Checkpoint order or prompt mismatch')
        if saved.get('score') != score(questions[item['id']], saved['raw'], prompt_plan['mode'], item['gold']):
            raise ValueError('Checkpoint score mismatch')
        if saved.get('band') != questions[item['id']]['band']:
            raise ValueError('Checkpoint band mismatch')
        if saved.get('finish_reason') not in {'stop', 'length'} or any(
                type(saved.get(k)) is not int or saved[k] < 0 for k in ('output_tokens', 'prompt_tokens')):
            raise ValueError('Invalid checkpoint telemetry')
        if saved['output_tokens'] > prompt_plan['generation']['max_new_tokens'] or saved['prompt_tokens'] + prompt_plan['generation']['max_new_tokens'] > prompt_plan['generation']['num_ctx']:
            raise ValueError('Checkpoint token budget mismatch')


def run(backend, mode, output, resume=False):
    suite, _ = load()
    prompt_plan = plan(mode)
    questions = {q['id']: q for q in suite['questions']}
    lock = json.loads((ROOT / 'data/v4_0_1/release.lock.json').read_text(encoding='utf-8'))
    identity = {'release': lock, 'prompt_plan_sha256': prompt_plan['prompt_plan_sha256'],
                'mode': mode, 'generation': prompt_plan['generation'], 'model': backend.identity()}
    run_id = digest(identity)
    result = {'schema_version': 4, 'run_id': run_id, 'identity': identity, 'records': [], 'complete': False}
    if output.exists():
        if not resume:
            raise ValueError('Output already exists; use explicit --resume')
        result = json.loads(output.read_text(encoding='utf-8'))
        if result.get('identity') != identity or result.get('run_id') != run_id:
            raise ValueError('Resume identity mismatch')
        validate_records(result['records'], prompt_plan, questions)
    elif resume:
        raise ValueError('No checkpoint to resume')
    for item in prompt_plan['items'][len(result['records']):]:
        generated = backend.generate(item, prompt_plan['generation'])
        if backend.identity() != identity['model']:
            raise ValueError('Model identity changed during generation')
        record = {key: item[key] for key in ('id', 'rotation', 'prompt_sha256')}
        record['band'] = questions[item['id']]['band']
        record.update(generated)
        record['score'] = score(questions[item['id']], record['raw'], mode, item['gold'])
        result['records'].append(record)
        validate_records(result['records'], prompt_plan, questions)
        result['summary'] = summarize(result['records'], mode)
        write_atomic(output, result)
    result['complete'] = True
    result['summary'] = summarize(result['records'], mode)
    write_atomic(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['free', 'mcq'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--gguf', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:11434')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.out.resolve()
    # Evaluation never writes into frozen evidence or source directories.
    for directory in ('manifests', 'manifests_qwen3', 'data', 'reproduce', 'bench', 'paper'):
        if output.is_relative_to(ROOT / directory):
            parser.error('Choose an output path outside historical evidence/source directories')
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        if output.exists() or args.resume:
            parser.error('Prompt export must use a new path')
        write_atomic(output, plan(args.mode))
    else:
        if not args.model or not args.gguf:
            parser.error('--model and --gguf are required for evaluation')
        run(Ollama(args.url, args.model, args.gguf), args.mode, output, args.resume)


if __name__ == '__main__':
    main()
