# Environment Setup (uv + WSL GPU)

## What is configured
- Python env managed by `uv` in `.venv`
- CUDA-enabled PyTorch installed (`torch==2.6.0+cu124`)
- Training dependencies installed (Transformers, Datasets, Accelerate, SentencePiece, MTEB)
- GPU/WSL diagnostics script added: `scripts/check_wsl_gpu.py`

## One-time setup
```bash
uv venv .venv
uv sync
```

## Run diagnostics
```bash
uv run python scripts/check_wsl_gpu.py
```

## Run a training sanity check
This validates mixed-precision training and reports tokens/sec on the best GPU (largest VRAM by default).

```bash
uv run python scripts/train_sanity.py --steps 80 --warmup 10
```

Pin a specific GPU if needed:

```bash
uv run python scripts/train_sanity.py --device-index 0
```

## v1 training scaffold (real data path)
Input file format (JSONL):
- pair mode: `{"anchor": "...", "positive": "..."}`
- or text mode: `{"text": "..."}` (self-supervised positives)

Train a tokenizer:
```bash
uv run python scripts/train_tokenizer.py --train-file data/sample_pairs.jsonl --output-dir artifacts/tokenizer-v1
```

Run v1 training:
```bash
uv run python scripts/train_v1.py \
  --train-file data/sample_pairs.jsonl \
  --tokenizer-path artifacts/tokenizer-v1 \
  --output-dir runs/v1 \
  --max-steps 200 \
  --save-every 100
```

Run with eval hooks:
```bash
uv run python scripts/train_v1.py \
  --train-file data/sample_pairs.jsonl \
  --eval-file data/sample_pairs.jsonl \
  --tokenizer-path artifacts/tokenizer-v1 \
  --output-dir runs/v1-eval
```

## Pull public datasets
```bash
uv run python scripts/download_public_data.py
```

Then train on pulled data:
```bash
uv run python scripts/train_tokenizer.py --train-file data/public/raw_code_text.jsonl --output-dir artifacts/tokenizer-v1
uv run python scripts/train_v1.py --train-file data/public/train_pairs.jsonl --eval-file data/public/eval_pairs.jsonl --tokenizer-path artifacts/tokenizer-v1 --output-dir runs/v1-public
```

Mine hard negatives and continue training:
```bash
uv run python scripts/mine_hard_negatives.py --input-file data/public/train_pairs.jsonl --checkpoint runs/v1-public/checkpoint-5000/training_state.pt --output-file data/public/train_pairs_hard.jsonl
uv run python scripts/train_v1.py --train-file data/public/train_pairs_hard.jsonl --eval-file data/public/eval_pairs.jsonl --tokenizer-path artifacts/tokenizer-v1 --output-dir runs/v1-public-hardneg
```

Export ONNX and benchmark:
```bash
uv run python scripts/export_onnx.py --checkpoint runs/v1-public-hardneg/checkpoint-10000/training_state.pt --output artifacts/model-v1.onnx --benchmark --batch-size 16 --seq-len 128
```

## Run end-to-end in tmux
This runs: sync -> data pull (if missing) -> tokenizer (if missing) -> base train -> hard-negative mining -> hard-negative train.

```bash
./scripts/run_v1_tmux.sh
```

Useful controls:
```bash
tmux attach -t emb-v1
tmux ls
tail -f runs/logs/emb-v1-*.log
```

Optional overrides:
```bash
SESSION_NAME=emb-v1-fast GPU_INDEX=0 MAX_STEPS_BASE=2000 MAX_STEPS_HARDNEG=2000 BATCH_SIZE=32 ./scripts/run_v1_tmux.sh
```

## Use NVIDIA SMI in WSL
In this environment, `nvidia-smi` lives at `/usr/lib/wsl/lib/nvidia-smi`.

Optional convenience in your shell profile:
```bash
export PATH="/usr/lib/wsl/lib:$PATH"
```

## Expected healthy output
- `is_wsl: true`
- `dxg_present: true`
- `torch.cuda_available: true`
- `torch.cuda_device_count >= 1`
- `nvidia_smi.ok: true`
