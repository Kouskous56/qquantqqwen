# Math50 V4 migration audit

Even unchanged stems used historical instructions, budgets and backends. None of these scores is a V4 experiment. Qwen3 original scorer provenance remains unknown.

Stage A recomputes canonical V3 scoring. Stage B applies V4 rules to OLD keys and OLD outputs; no question edits are mixed into this difference. Stage C audits only the 32 unchanged stems with new aliases; it is still not a V4 run.

| Source | Rep | Stored | V3 A /50 | V4 rules B /50 | Gained | Lost | C /32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RUN_V31_05B_S0.json | 1 | 17 | 18 | 18 | 0 | 0 | 15 |
| RUN_V31_05B_S0.json | 2 | 17 | 18 | 18 | 0 | 0 | 15 |
| RUN_V31_05B_S0.json | 3 | 17 | 18 | 18 | 0 | 0 | 15 |
| RUN_V31_05B_S30.json | 1 | 12 | 11 | 11 | 0 | 0 | 9 |
| RUN_V31_05B_S30.json | 2 | 12 | 11 | 11 | 0 | 0 | 9 |
| RUN_V31_05B_S30.json | 3 | 12 | 11 | 11 | 0 | 0 | 9 |
| RUN_V31_15B_S0.json | 1 | 39 | 40 | 40 | 0 | 0 | 26 |
| RUN_V31_15B_S0.json | 2 | 39 | 40 | 40 | 0 | 0 | 26 |
| RUN_V31_15B_S0.json | 3 | 39 | 40 | 40 | 0 | 0 | 26 |
| RUN_V31_15B_S30.json | 1 | 33 | 33 | 32 | 0 | 1 | 23 |
| RUN_V31_15B_S30.json | 2 | 33 | 33 | 32 | 0 | 1 | 23 |
| RUN_V31_15B_S30.json | 3 | 33 | 33 | 32 | 0 | 1 | 23 |
| RUN_V31_3B_S0.json | 1 | 40 | 42 | 41 | 0 | 1 | 27 |
| RUN_V31_3B_S0.json | 2 | 40 | 42 | 41 | 0 | 1 | 27 |
| RUN_V31_3B_S0.json | 3 | 40 | 42 | 41 | 0 | 1 | 27 |
| RUN_V31_3B_S30.json | 1 | 39 | 40 | 39 | 0 | 1 | 27 |
| RUN_V31_3B_S30.json | 2 | 39 | 40 | 39 | 0 | 1 | 27 |
| RUN_V31_3B_S30.json | 3 | 39 | 40 | 39 | 0 | 1 | 27 |
| RUN_QWEN3_TYPED_DENSE2.json | 1 | 43 | 43 | 42 | 0 | 1 | 28 |
| RUN_QWEN3_TYPED_EORA.json | 1 | 42 | 42 | 41 | 0 | 1 | 28 |
| RUN_QWEN3_TYPED_S20C4.json | 1 | 43 | 43 | 42 | 0 | 1 | 28 |
| RUN_QWEN3_TYPED_S20MIXED.json | 1 | 43 | 43 | 42 | 0 | 1 | 28 |
| RUN_QWEN3_TYPED_S30C4.json | 1 | 40 | 40 | 40 | 0 | 0 | 28 |
| RUN_QWEN3_TYPED_S30MIXED.json | 1 | 41 | 41 | 40 | 0 | 1 | 27 |

Changed stems excluded from stage C: h17, h23, h24, h27, h29, h30, o33, o34, o35, o38, o39, a41, a42, a44, a45, a48, a49, a50

The companion JSON contains full source hashes and every changed judgment with its original public raw output. Repeated deterministic generations are not independent samples. Score losses here establish a parsing-policy difference, not a model-quality regression.
