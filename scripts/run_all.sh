#!/usr/bin/env bash
set -euo pipefail

python -m scripts.train_tokenizer "$@"
python -m scripts.prepare_data "$@"
python -m train.train
python -m finetune.sft
python -m finetune.dpo
python -m eval.run_evals
python -m inference.server --model checkpoints/dpo --port 8000
