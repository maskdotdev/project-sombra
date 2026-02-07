#!/usr/bin/env bash
set -euo pipefail

# Executed inside tmux by run_v1_tmux.sh.

GPU_INDEX="${GPU_INDEX:-0}"
MAX_STEPS_BASE="${MAX_STEPS_BASE:-5000}"
MAX_STEPS_HARDNEG="${MAX_STEPS_HARDNEG:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOKENIZER_DIR="${TOKENIZER_DIR:-artifacts/tokenizer-v1}"
RUN_BASE_DIR="${RUN_BASE_DIR:-runs/v1-public}"
RUN_HN_DIR="${RUN_HN_DIR:-runs/v1-public-hardneg}"
TRAIN_PAIRS="${TRAIN_PAIRS:-data/public/train_pairs.jsonl}"
EVAL_PAIRS="${EVAL_PAIRS:-data/public/eval_pairs.jsonl}"
RAW_TEXT="${RAW_TEXT:-data/public/raw_code_text.jsonl}"
HARDNEG_PAIRS="${HARDNEG_PAIRS:-data/public/train_pairs_hard.jsonl}"

echo "[1/6] sync env"
uv sync

echo "[2/6] pull public data if missing"
if [ ! -f "$TRAIN_PAIRS" ] || [ ! -f "$EVAL_PAIRS" ] || [ ! -f "$RAW_TEXT" ]; then
  uv run python scripts/download_public_data.py
fi

echo "[3/6] train tokenizer if missing"
if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ]; then
  uv run python scripts/train_tokenizer.py --train-file "$RAW_TEXT" --output-dir "$TOKENIZER_DIR"
fi

echo "[3b/6] validate tokenizer unk token"
if ! TOKENIZER_DIR="$TOKENIZER_DIR" uv run python - <<'PY'
import os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_DIR"], use_fast=True)
ok = tok.unk_token_id is not None
print(f"unk_token_id={tok.unk_token_id}")
raise SystemExit(0 if ok else 1)
PY
then
  echo "Tokenizer missing unk token fallback; retraining tokenizer"
  uv run python scripts/train_tokenizer.py --train-file "$RAW_TEXT" --output-dir "$TOKENIZER_DIR"
fi

echo "[4/6] base train"
uv run python scripts/train_v1.py \
  --train-file "$TRAIN_PAIRS" \
  --eval-file "$EVAL_PAIRS" \
  --tokenizer-path "$TOKENIZER_DIR" \
  --output-dir "$RUN_BASE_DIR" \
  --max-steps "$MAX_STEPS_BASE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device-index "$GPU_INDEX" \
  --eval-every 500 \
  --save-every 500

LATEST_BASE_CKPT=$(ls -d "$RUN_BASE_DIR"/checkpoint-* | sort -V | tail -n 1)

echo "[5/6] mine hard negatives"
uv run python scripts/mine_hard_negatives.py \
  --input-file "$TRAIN_PAIRS" \
  --checkpoint "$LATEST_BASE_CKPT/training_state.pt" \
  --tokenizer-path "$TOKENIZER_DIR" \
  --output-file "$HARDNEG_PAIRS" \
  --max-samples 20000 \
  --batch-size 128 \
  --device-index "$GPU_INDEX"

echo "[6/6] hard-negative train"
uv run python scripts/train_v1.py \
  --train-file "$HARDNEG_PAIRS" \
  --eval-file "$EVAL_PAIRS" \
  --tokenizer-path "$TOKENIZER_DIR" \
  --output-dir "$RUN_HN_DIR" \
  --max-steps "$MAX_STEPS_HARDNEG" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device-index "$GPU_INDEX" \
  --hard-neg-weight 0.3 \
  --eval-every 500 \
  --save-every 500

echo "done"
