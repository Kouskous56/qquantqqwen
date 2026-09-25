# third_party_adapters — our modifications of upstream tools
`sparsegpt_qwen.py` is a Qwen2 port of IST-DASLab/SparseGPT
(base commit `147d2159dc4f3e9f73e47b32c04d7b3708f44436`), adding:
Qwen2ForCausalLM loading, AutoTokenizer, manually computed RoPE
position embeddings for direct per-layer calls under transformers 5.x,
and a Salesforce/wikitext dataset-ID fix.

Upstream license applies to the derived portions; our port changes are MIT.
To reproduce: clone the base commit, copy this file alongside it as `qwen.py`.
