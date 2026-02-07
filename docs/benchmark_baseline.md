# Baseline Benchmark Snapshot

This is the current baseline after first full training pass.

## Quality (2,000 eval pairs)
- `recall@1`: `0.149`
- `recall@5`: `0.266`
- `recall@10`: `0.331`
- `mrr@10`: `0.2017`

## Inference throughput
- GPU ONNX (FP32 model): `~1609 samples/s`
- CPU ONNX (INT8 model): `~94 samples/s`
- CPU ONNX (FP32 model): `~53 samples/s`

## Artifact sizes
- `artifacts/model-v1-candidate.onnx`: `161 MB`
- `artifacts/model-v1-candidate.int8.onnx`: `41 MB`

## Is this good enough?
Short answer: this is a solid technical baseline, but likely not product-ready quality yet for strong code search UX.

- Throughput is strong, especially on GPU.
- Retrieval quality is usable but not yet strong enough for high-confidence top-k relevance in many real codebases.
- The right next move is not just scaling model size; it is improving pair quality and hard-negative quality.

## Why quality is currently limited
- Public mixed datasets are noisy and heterogeneous.
- First hard-negative round likely included false negatives.
- Limited domain adaptation to your own code style/repo structure.
- Single-pass training schedule; no iterative mine-train-mine loop yet.
