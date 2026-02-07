# Next Iterations

## Priority 1 (highest ROI)
- [ ] Build higher-quality training pairs from real repos (docstring->function, signature->body, issue/title->patch when available).
- [ ] Add false-negative filtering in hard-negative mining (same symbol/file/repo family checks).
- [ ] Run iterative curriculum: base train -> mine negatives -> short retrain -> mine again.
- [ ] Keep a strict held-out benchmark split by repository (avoid leakage and optimistic metrics).

## Priority 2 (quality + speed)
- [ ] Train a slightly larger teacher (for quality), then distill into small fast student for serving.
- [ ] Evaluate `max_seq_len=128` vs `256` tradeoff against latency and recall.
- [ ] Try embedding dims `256` vs `384` to improve CPU throughput.

## Priority 3 (serving and product fit)
- [ ] Add ANN operating-point sweep (`efSearch`, latency, recall curves).
- [ ] Build end-to-end retrieval eval on your real query logs/tasks.
- [ ] Add reranker only for top-N if needed (keep first stage fast).

## Suggested immediate experiment pack
1. `E1`: Better negatives only (same model size), 3k-5k steps.
2. `E2`: Domain-focused data mix, 5k-10k steps.
3. `E3`: Distill to CPU-first student and compare recall drop vs throughput gain.

## Exit criteria for v1 ship candidate
- `recall@10` improves materially from `0.331` on held-out repo split.
- P95 encode latency meets target SLO on deployment hardware.
- Retrieval quality validated on representative internal code-search tasks.
