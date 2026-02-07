# Progress Log

## 2026-02-06
- Bootstrapped from-scratch code embedding pipeline with `uv` and WSL2 CUDA validation.
- Confirmed GPU visibility in WSL: RTX 3090 Ti (24 GB) + RTX 4060 Ti (8 GB).
- Added tokenizer training, v1 model training, hard-negative mining, ONNX export, and retrieval benchmark scripts.
- Pulled public starter datasets and normalized into JSONL.
- Completed full base + hard-negative training pipeline in tmux.
- Kept best run artifacts and cleaned intermediate checkpoints/logs.

### Best run summary
- Base run: `runs/v1-public/checkpoint-5000`
- Best offline retrieval on current eval set:
  - `recall@1 = 0.149`
  - `recall@5 = 0.266`
  - `recall@10 = 0.331`

### Serving summary
- GPU ONNX throughput (batch benchmark): `~1609 samples/s`
- CPU INT8 throughput (batch benchmark): `~94 samples/s`
- ANN search is fast and not the bottleneck.
