# Embedding Project Docs

This directory is the working notebook for model quality, performance, and iteration decisions.

## Files
- `docs/progress_log.md`: chronological run notes and outcomes.
- `docs/benchmark_baseline.md`: current benchmark snapshot and interpretation.
- `docs/next_iterations.md`: prioritized improvement plan and experiment matrix.

## How to use this folder
1. After each training run, append one short entry to `docs/progress_log.md`.
2. If metrics change, update `docs/benchmark_baseline.md` with new numbers.
3. Move completed items in `docs/next_iterations.md` from pending to done.

## Current best artifacts
- Checkpoint: `runs/v1-public/checkpoint-5000`
- Tokenizer: `artifacts/tokenizer-v1`
- ONNX FP32: `artifacts/model-v1-candidate.onnx`
- ONNX INT8: `artifacts/model-v1-candidate.int8.onnx`
- Reports:
  - `runs/retrieval_report_gpu.json`
  - `runs/retrieval_report_cpu_fp32.json`
  - `runs/retrieval_report_cpu_int8.json`
