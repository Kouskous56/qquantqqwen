# IFEval matched hybrid: dense-Q4 vs s30+Q4 (541 prompts, audit 541 unique)
| Model | prompt-strict | prompt-loose | inst-strict | inst-loose |
|---|---|---|---|---|
| mine-Q4 (dense) | 0.6007 | 0.6414 | 0.6882 | 0.7242 |
| s30-Q4 (hybrid) | 0.5656 | 0.6137 | 0.6595 | 0.7002 |
| Retention | 94% | 96% | 96% | 97% |
Ket luan: hybrid Wanda-s30 + Q4 GIU instruction following (~94-97%).
Deeper: instruction compliance robust hon nhieu so voi GSM/math-free cua 0.5B.
s30-Q4 la deployment candidate that: 1.9GB, ~80 tok/s, IFEval ~0.57 strict.
