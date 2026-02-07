# Public Data + Training Approach

## Objective
Train an English-first code embedding model from scratch with low latency and high throughput, while using publicly available data to bootstrap quickly.

## Approach We Are Taking
- Use a two-track data strategy:
  - **Pair data** (`anchor`, `positive`) for contrastive embedding training.
  - **Raw code text** (`text`) for tokenizer training and optional representation pretraining.
- Start with small-to-medium public subsets for fast iteration, then scale once metrics and throughput are stable.
- Keep model size moderate (latency-first): encoder-only transformer, 384-512 output embedding dimension.

## Current Model Shape (v1)
- Architecture: BERT-style encoder initialized from scratch.
- Default config in `scripts/train_v1.py`:
  - hidden size: `512`
  - layers: `8`
  - attention heads: `8`
  - FFN size: `2048`
  - max sequence length: `256`
  - embedding output dim: `384`
- Pooling: masked mean pooling over token states, then linear projection + L2 normalization.
- Loss: symmetric in-batch InfoNCE, with optional hard-negative term when `negative` is present.

## Public Sources Included
- **CodeXGLUE code-to-text** (`google/code_x_glue_ct_code_to_text`): docstring-to-code pairs for Python, JavaScript, Go, Java, Ruby, PHP.
- **LeetCode corpus** (`greengerong/leetcode`): problem statement to code solution pairs (we currently use Python solutions).
- **MBPP** (`mbpp`): task text with reference code pairs.
- **HumanEval** (`openai_humaneval`): prompt + canonical solution (primarily eval signal).
- **Raw code/text corpora**:
  - `code-rag-bench/programming-solutions`
  - `code-rag-bench/stackoverflow-posts`
  - `code-rag-bench/online-tutorials`

Notes:
- The pull script writes per-source success/failure details in `data/public/manifest.json`.
- Licensing/compliance checks should be enforced before production use.

## Data Files Produced
All outputs are written to `data/public`:
- `data/public/train_pairs.jsonl`
- `data/public/eval_pairs.jsonl`
- `data/public/raw_code_text.jsonl`
- `data/public/manifest.json`
- optional hard-negative train set: `data/public/train_pairs_hard.jsonl`

Record format:
- Pair rows: `{"anchor": "...", "positive": "...", "source": "...", "split": "..."}`
- Raw rows: `{"text": "...", "source": "...", "split": "..."}`

## Commands
Pull public data:

```bash
uv run python scripts/download_public_data.py
```

Current pulled starter set (already downloaded in this workspace):
- train pairs: `10,874`
- eval pairs: `2,004`
- raw text/code rows: `6,000`

Adjust pull sizes:

```bash
uv run python scripts/download_public_data.py \
  --codesearchnet-train-per-lang 5000 \
  --codesearchnet-eval-per-lang 800 \
  --raw-code 50000
```

Train tokenizer on raw text:

```bash
uv run python scripts/train_tokenizer.py \
  --train-file data/public/raw_code_text.jsonl \
  --output-dir artifacts/tokenizer-v1 \
  --vocab-size 32000
```

Run v1 embedding training on public pair data:

```bash
uv run python scripts/train_v1.py \
  --train-file data/public/train_pairs.jsonl \
  --eval-file data/public/eval_pairs.jsonl \
  --tokenizer-path artifacts/tokenizer-v1 \
  --output-dir runs/v1-public \
  --max-steps 20000 \
  --save-every 1000 \
  --eval-every 500
```

Mine hard negatives from a checkpoint and continue training:

```bash
uv run python scripts/mine_hard_negatives.py \
  --input-file data/public/train_pairs.jsonl \
  --checkpoint runs/v1-public/checkpoint-5000/training_state.pt \
  --output-file data/public/train_pairs_hard.jsonl

uv run python scripts/train_v1.py \
  --train-file data/public/train_pairs_hard.jsonl \
  --eval-file data/public/eval_pairs.jsonl \
  --tokenizer-path artifacts/tokenizer-v1 \
  --output-dir runs/v1-public-hardneg \
  --hard-neg-weight 0.3
```

Export ONNX for serving and benchmark latency:

```bash
uv run python scripts/export_onnx.py \
  --checkpoint runs/v1-public-hardneg/checkpoint-10000/training_state.pt \
  --output artifacts/model-v1.onnx \
  --benchmark \
  --batch-size 16 \
  --seq-len 128
```

## Why This Setup Works
- Pair data gives direct retrieval signal for embeddings.
- Raw code data improves tokenizer coverage and representation quality.
- The training scaffold is GPU-aware and defaults to highest-VRAM GPU (3090 Ti in this environment).
- Checkpoint + recall hooks make it easy to pick the best model before serving optimization.

## Next Scaling Steps
- Add iterative hard-negative refresh every few epochs (mine -> continue train -> mine again).
- Increase corpus size and language balance after baseline quality is stable.
- Add ANN evaluation loop (HNSW/FAISS) for latency/recall tradeoff tracking.
- Export ONNX + int8 quantized model for production throughput.
