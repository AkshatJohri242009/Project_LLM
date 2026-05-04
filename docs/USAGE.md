# Usage

This project is a small GPT-style LLM stack. The code is real; the local checkpoint included from smoke tests is only for proving the pipeline works, not for quality.

## Mental Model

1. `scripts.train_tokenizer` learns a BPE tokenizer and writes `tokenizer/`.
2. `scripts.prepare_data` streams text, filters/deduplicates it, tokenizes it, and writes packed `.bin` shards.
3. `train.train` reads those shards and trains the decoder-only transformer.
4. `finetune.sft` turns the base model into a chat model with supervised examples and LoRA.
5. `finetune.dpo` aligns the SFT model with chosen/rejected preference pairs.
6. `inference.generate` loads a checkpoint and samples text with a KV cache.
7. `inference.server` exposes `/v1/chat/completions` and `/v1/models`.
8. `eval.run_evals` runs `lm-eval` and writes JSON reports.

## Quick Local Smoke Test

```powershell
cd D:\it\llm-repo
python -m pytest -q
python -m scripts.train_tokenizer --smoke --sample-bytes 10000 --vocab-size 512
python -m scripts.prepare_data --smoke --seq-len 32 --max-sequences 8
python -m train.train --max-steps 1 --smoke --batch-size 1 --seq-len 16 --checkpoint-dir checkpoints/smoke_tokenized
python -m inference.generate --model checkpoints/smoke_tokenized/step_1 --prompt "explain gravity to a 5 year old" --max-new-tokens 20
```

The output will be nonsense because it is a tiny one-step model. Success means the pipeline runs.

## Full Training Shape

```powershell
python -m scripts.train_tokenizer
python -m scripts.prepare_data
torchrun --nproc_per_node=1 -m train.train
python -m finetune.sft --base-model checkpoints/step_100000
python -m finetune.dpo --model checkpoints/sft
python -m eval.run_evals --model checkpoints/dpo
```

Use more `torchrun` processes on a multi-GPU machine. Full training downloads large datasets and needs serious GPU time.

## Run The API Server

```powershell
python -m inference.server --model checkpoints/dpo --port 8000
```

Test it:

```powershell
$body = @{
  model = "scratch"
  messages = @(@{ role = "user"; content = "explain gravity to a 5 year old" })
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body
```

## Upload To GitHub

Generated datasets, checkpoints, tokenizer files, eval outputs, and caches are ignored by `.gitignore`, so the repository stays small.

```powershell
cd D:\it\llm-repo
git init
git add .
git commit -m "Initial scratch LLM codebase"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Free Local Disk Space

Preview cleanup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\clean_artifacts.ps1 -WhatIf
```

Actually remove generated checkpoints, data shards, eval outputs, and caches:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\clean_artifacts.ps1
```

Also remove tokenizer artifacts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\clean_artifacts.ps1 -IncludeTokenizer
```
