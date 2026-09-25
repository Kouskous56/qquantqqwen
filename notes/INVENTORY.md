# Inventory Qwen Edge AI - RTX 4050 6GB / RAM 16GB

## A. Công cụ
- Ollama 0.34.3 | server chạy, KEEP_ALIVE=1m | models 100% GPU
- Python 3.11.9 | torch 2.9.0+cu129 cuda=True | transformers 5.17.0
- accelerate 1.15.0 | datasets 5.0.1 | peft 0.21.0 | llama-cpp-python 0.3.35
- huggingface_hub 1.32.0 + hf CLI | git 2.55.0 | cmake 4.4.3 | MSVC 14.44
- LongPathsEnabled = 1
- tools/wanda | bench/* | scripts/* (đầy đủ)

## B. Mô hình Ollama
| Model | Size | Params | Context | Quant | Ghi chú |
|---|---|---|---|---|---|
| qwen2.5:0.5b | 397MB | 494M | 32K | Q4_K_M | baseline tiny |
| qwen2.5:1.5b | 986MB | 1.5B | 32K | Q4_K_M | - |
| qwen2.5:3b | 1.9GB | 3.1B | 32K | Q4_K_M | base chính |
| qwen3:4b | 2.5GB | 4.0B | 256K | Q4_K_M | thinking mode |
| qwen3b-mine-q4 | 1.9GB | 3.1B | 32K | Q4_K_M | tự quant, chatbot chính |
| s20r-q4 | 481MB | 0.5B | 4K | Q4_K_M | hybrid s20+LoRA+Q4 |

## C. Mô hình Hugging Face (D:\qwen\models)
- 0.5B: Q4 468MB (tải sẵn) + FP16 942MB + tự quant F16/Q8/Q4/Q3
- 1.5B Q4 1065MB | 3B Q4 2007MB | 3B FP16 5.9GB + tự quant Q8/Q5/Q4/Q3
- Pruned: s20/s30/s50/2:4 + adapters LoRA (s20, s30-wiki, s30-gsm8k) + masks
- Kiến trúc 0.5B: Qwen2ForCausalLM, 24 layers, hidden 896, vocab 151936
