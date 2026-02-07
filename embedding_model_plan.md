# Code Embedding Model Plan (From Scratch)

## Goal
Train an English-only code embedding model from scratch with very low latency and high throughput for semantic code search/retrieval.

## Current Known Constraints
- Domain: code
- Language scope: English only
- Data readiness: starting from zero
- Serving target: very low latency, high throughput
- Environment note: WSL2 with CUDA toolkit (`nvcc 12.4`) present; exact GPU model/VRAM still to be confirmed from host `nvidia-smi`.

## Target System (Default)
Use this default until exact GPU specs are confirmed:
- Model size: 60M to 120M parameters
- Embedding dimension: 384 or 512
- Max sequence length: 256 (increase only if budget allows)
- Encoder-only transformer (12 layers)
- Inference format: ONNX + int8 quantization
- ANN index: HNSW

This size range is a strong fit for low-latency serving while still achieving strong retrieval quality.

## Success Criteria
- Retrieval quality:
  - `Recall@10` and `MRR` on held-out code search benchmark
  - Improvement over baseline (BM25 and/or existing embedding model)
- Performance:
  - p95 embedding latency within service budget
  - Throughput target achieved at expected batch size
  - Stable memory footprint for inference and ANN index

## High-Level Architecture
1. Tokenizer from scratch (SentencePiece Unigram/BPE)
2. Encoder pretraining on raw code corpus (MLM or RTD objective)
3. Contrastive embedding training (InfoNCE with in-batch negatives)
4. Hard-negative mining refresh loop
5. Export/quantize model for production serving
6. Build ANN retrieval stack and tune latency-quality tradeoff

## Data Plan (Start Here)

### 1) Source Data
- Collect permissive-license repositories (MIT/BSD/Apache)
- Include:
  - source files
  - README/docs
  - docstrings/comments
- Prioritize languages (initial suggestion): Python, TypeScript, JavaScript, Go, Java, Rust

### 2) Cleaning + Normalization
- Deduplicate exact and near-duplicate files/functions
- Remove generated/vendor/minified files
- Filter tiny/empty/garbled snippets
- Keep metadata: repo, language, file path, symbol/function boundaries

### 3) Training Pair Construction
Construct positive pairs automatically:
- `(docstring/comment -> function/class body)`
- `(function signature/name -> function body)`
- `(README section -> relevant file/snippet)`

Negatives:
- In-batch negatives by default
- Hard negatives from same repo/language during later rounds

## Training Stages

### Stage A: Tokenizer
- Train SentencePiece tokenizer on cleaned code corpus
- Suggested vocab: 32k to 50k
- Validate tokenization quality on representative code samples

### Stage B: Encoder Pretraining (Representation Learning)
- Objective: MLM (or RTD)
- Suggested model: 12-layer encoder, hidden 512-768, 8-12 heads
- Sequence length: 256 initially
- Save checkpoints regularly and track validation loss/perplexity proxy

### Stage C: Embedding Training (Retrieval Objective)
- Objective: contrastive InfoNCE
- Pooling: mean pooling over final hidden states
- Output normalization: L2 normalization
- Temperature: tune in range `0.02` to `0.07`
- Large effective batch (use grad accumulation as needed)

### Stage D: Hard-Negative Mining Loop
- Build interim ANN index on checkpoint embeddings
- Mine nearest non-positive examples as hard negatives
- Continue contrastive training with mixed negatives

## Evaluation Plan

### Offline Quality
- Benchmark sets:
  - held-out internal repo pairs
  - external code search benchmark (for sanity)
- Metrics:
  - `Recall@1/10`
  - `MRR`
  - `nDCG@10`

### Online/Serving Quality
- Embedding latency p50/p95
- Throughput at realistic concurrent load
- ANN recall vs latency under varying `efSearch`

## Serving Plan (Latency/Throughput First)
1. Export to ONNX
2. Apply int8 quantization
3. Use batch inference service (dynamic batching)
4. Build HNSW index with tuned params:
   - start with `M=16`, `efConstruction=200`
   - tune `efSearch` for latency/recall target
5. Cache popular query embeddings and hot ANN results

## Implementation Timeline (6 Weeks)

### Week 1: Foundations
- Finalize data licenses and language mix
- Implement ingestion, cleaning, dedup pipeline
- Produce first curated corpus snapshot

### Week 2: Tokenizer + Baseline
- Train tokenizer
- Prepare train/val splits
- Create baseline retrieval benchmark harness

### Week 3-4: Pretraining
- Train encoder with MLM/RTD
- Track training stability and checkpoint quality
- Run periodic probe retrieval evaluations

### Week 5: Contrastive Training
- Build positive/negative pair datasets
- Train embedding objective
- Start hard-negative mining refresh

### Week 6: Optimize + Ship Candidate
- ONNX export + int8 quantization
- Build/tune HNSW index
- Final benchmark report (quality + latency + throughput)
- Select release candidate model

## Risks and Mitigations
- Unknown GPU budget -> start with smaller model and scale only after profiling
- Data quality noise -> strict filtering + hard-negative curriculum
- High latency -> enforce small embedding dimension and int8 path early
- Overfitting to narrow code style -> diversify repos/languages and hold-out by repository

## Immediate Next Actions
1. Confirm GPU inventory from host machine (`nvidia-smi` output)
2. Pick initial language subset for v1 corpus
3. Set hard latency SLO (example: p95 <= 20 ms per snippet at batch=32)
4. Start data ingestion and dedup pipeline implementation

---

If GPU details are provided, this plan should be updated with exact model width, batch size, gradient accumulation, and training-time estimates.
