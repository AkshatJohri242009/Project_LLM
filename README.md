# scratch-llm

Complete from-scratch GPT-style LLM codebase: tokenizer training, streaming data prep, pretraining, LoRA SFT, DPO, inference, OpenAI-compatible serving, and evals.

For a fuller runbook, including GitHub upload and cleanup commands, see [`docs/USAGE.md`](docs/USAGE.md).

## Setup

```bash
cd llm
python -m pip install -e .
```

Heavy packages such as `vllm` and `bitsandbytes` may require Linux/CUDA-specific wheels. The core model and smoke tests run without downloading model weights.

## Tokenizer

```bash
python -m scripts.train_tokenizer
python -m scripts.train_tokenizer --smoke --sample-bytes 200000
```

The script streams `HuggingFaceFW/fineweb` `sample-10BT`, falls back to `wikitext-103`, and saves a 32k BPE tokenizer to `tokenizer/`.

## Data

```bash
python -m scripts.prepare_data
python -m scripts.prepare_data --smoke --max-sequences 64
```

Data prep filters language metadata, drops low-quality documents, deduplicates with MinHash LSH, tokenizes, packs 2048-token sequences, and writes uint16 `train_*.bin` / `val_*.bin` shards.

## Pretraining

```bash
python -m train.train
torchrun --nproc_per_node=8 -m train.train
```

If no GPU or no shards are present, `python -m train.train` uses a 2-layer CPU smoke model and runs 100 synthetic steps. Full defaults are in `configs/base.yaml`.

## SFT And DPO

```bash
python -m finetune.sft --base-model checkpoints/step_100000
python -m finetune.dpo --model checkpoints/sft
```

SFT uses ChatML and masks system/user labels. DPO uses TRL `DPOTrainer` when available and falls back to a compact manual trainer.

## Inference

```bash
python -m inference.generate --model checkpoints/dpo --prompt "explain gravity to a 5 year old"
python -m inference.generate --model checkpoints/dpo --quantize
```

## Server

```bash
python -m inference.server --model checkpoints/dpo --port 8000
curl http://localhost:8000/v1/models
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"scratch","messages":[{"role":"user","content":"explain gravity to a 5 year old"}]}'
```

## Evaluation

```bash
python -m eval.run_evals --model checkpoints/dpo
```

Runs `hellaswag`, `arc_easy`, `arc_challenge`, `piqa`, and `winogrande` through `lm-eval`, then saves JSON reports under `eval/results/`.

## One-Shot

```bash
bash scripts/run_all.sh
```

For quick local validation, run:

```bash
pytest
python -m train.train --max-steps 2 --smoke
```

Generated data, tokenizer files, checkpoints, and eval outputs are ignored by Git so the repository can be uploaded without the heavy artifacts.
