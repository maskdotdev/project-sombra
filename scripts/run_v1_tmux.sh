#!/usr/bin/env bash
set -euo pipefail

# Long-running training pipeline for tmux usage.
# Override defaults with environment variables if needed.

SESSION_NAME="${SESSION_NAME:-emb-v1}"
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
LOG_DIR="${LOG_DIR:-runs/logs}"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/${SESSION_NAME}-${TS}.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi
chmod +x scripts/run_v1_pipeline.sh

tmux new-session -d -s "$SESSION_NAME" \
  "GPU_INDEX='$GPU_INDEX' MAX_STEPS_BASE='$MAX_STEPS_BASE' MAX_STEPS_HARDNEG='$MAX_STEPS_HARDNEG' BATCH_SIZE='$BATCH_SIZE' NUM_WORKERS='$NUM_WORKERS' TOKENIZER_DIR='$TOKENIZER_DIR' RUN_BASE_DIR='$RUN_BASE_DIR' RUN_HN_DIR='$RUN_HN_DIR' TRAIN_PAIRS='$TRAIN_PAIRS' EVAL_PAIRS='$EVAL_PAIRS' RAW_TEXT='$RAW_TEXT' HARDNEG_PAIRS='$HARDNEG_PAIRS' bash scripts/run_v1_pipeline.sh |& tee '$LOG_FILE'"

echo "Started tmux session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Tail log: tail -f $LOG_FILE"
