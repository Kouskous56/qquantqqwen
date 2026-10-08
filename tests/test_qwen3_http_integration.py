"""Real HTTP transport and real artifact hashing; synthetic inference fixture."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from bench.qwen3 import common


def test_http_resume_and_alias_retarget(tmp_path):
    blob = tmp_path/'weights.gguf'
    blob.write_bytes(b'HTTP integration fixture, NOT a language model')
    blob_hash = common.sha256(blob)
    state = {'digest': 'a'*64, 'calls': 0, 'interrupt': True}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def reply(self, value, status=200):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())
        def do_GET(self):
            self.reply({'models': [{'name': 'fixture:latest', 'digest': state['digest']}]} if self.path == '/api/tags' else {'version': 'HTTP-fixture'})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/api/show':
                self.reply({'modelfile': 'FROM /blobs/sha256-' + blob_hash, 'template': 'fixture-template'})
            elif self.path == '/api/chat':
                assert body['model'] == 'fixture'
                state['calls'] += 1
                if state['calls'] == 2 and state['interrupt']:
                    self.reply({'error': 'simulated server interruption'}, 503)
                else:
                    self.reply({'done': True, 'message': {'content': '#### 2'}})
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    except OSError as exc:
        pytest.skip('Loopback sockets unavailable: ' + str(exc))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        args = argparse.Namespace(model='fixture', endpoint=f'http://127.0.0.1:{server.server_port}', tag='fixture', gguf=str(blob))
        items = [(i, {'question': '1+1?', 'answer': '#### 2'}) for i in range(3)]
        checkpoint = tmp_path/'checkpoint.json'
        with pytest.raises(Exception, match='503'):
            common.evaluate(args, items, 'gsm', checkpoint, __file__, 32)
        assert json.loads(checkpoint.read_text())['done'] == 1
        state['interrupt'] = False
        details, _, _ = common.evaluate(args, items, 'gsm', checkpoint, __file__, 32)
        assert len(details) == 3 and state['calls'] == 4
        before = checkpoint.read_bytes()
        state['digest'] = 'b'*64
        with pytest.raises(ValueError, match='different or legacy'):
            common.evaluate(args, items, 'gsm', checkpoint, __file__, 32)
        assert state['calls'] == 4 and checkpoint.read_bytes() == before
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
