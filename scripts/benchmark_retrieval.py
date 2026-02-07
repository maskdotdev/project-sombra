from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import hnswlib
import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark embedding retrieval quality and ANN latency.")
    p.add_argument("--onnx", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--report", default="runs/retrieval_report.json")
    return p.parse_args()


def encode_texts(
    sess: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int,
    max_seq_len: int,
) -> tuple[np.ndarray, list[float]]:
    out: list[np.ndarray] = []
    batch_times: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        toks = tokenizer(batch, truncation=True, max_length=max_seq_len, padding=True, return_tensors="np")
        t0 = time.perf_counter()
        emb = sess.run(None, {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]})[0]
        batch_times.append(time.perf_counter() - t0)
        out.append(emb)
    return np.concatenate(out, axis=0), batch_times


def recall_at(labels: np.ndarray, k: int) -> float:
    n = labels.shape[0]
    gt = np.arange(n)[:, None]
    return float((labels[:, :k] == gt).any(axis=1).mean())


def mrr_at(labels: np.ndarray, k: int) -> float:
    n = labels.shape[0]
    gt = np.arange(n)
    rr = np.zeros(n, dtype=np.float32)
    top = labels[:, :k]
    for i in range(n):
        hits = np.where(top[i] == gt[i])[0]
        if hits.size:
            rr[i] = 1.0 / float(hits[0] + 1)
    return float(rr.mean())


def main() -> int:
    args = parse_args()
    providers = ["CPUExecutionProvider"] if args.cpu_only else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    ds = load_dataset("json", data_files=args.pairs_file, split="train")
    if not {"anchor", "positive"}.issubset(set(ds.column_names)):
        raise ValueError("pairs file must have anchor and positive fields")
    n = min(args.max_samples, len(ds))
    ds = ds.select(range(n))

    anchors = [str(x) for x in ds["anchor"]]
    positives = [str(x) for x in ds["positive"]]

    q_emb, q_times = encode_texts(sess, tokenizer, anchors, args.batch_size, args.max_seq_len)
    c_emb, c_times = encode_texts(sess, tokenizer, positives, args.batch_size, args.max_seq_len)

    dim = c_emb.shape[1]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    idx.add_items(c_emb, np.arange(n))
    idx.set_ef(max(args.top_k, 50))

    t0 = time.perf_counter()
    labels, _ = idx.knn_query(q_emb, k=args.top_k)
    search_elapsed = time.perf_counter() - t0

    report = {
        "onnx": args.onnx,
        "providers": sess.get_providers(),
        "samples": n,
        "embedding_dim": int(dim),
        "metrics": {
            "recall@1": recall_at(labels, 1),
            "recall@5": recall_at(labels, 5),
            "recall@10": recall_at(labels, min(10, args.top_k)),
            "mrr@10": mrr_at(labels, min(10, args.top_k)),
        },
        "latency": {
            "encode_ms_p50": float(np.percentile(np.array(q_times + c_times) * 1000.0, 50)),
            "encode_ms_p95": float(np.percentile(np.array(q_times + c_times) * 1000.0, 95)),
            "encode_samples_per_sec": float((2 * n) / max(sum(q_times) + sum(c_times), 1e-9)),
            "ann_ms_total": float(search_elapsed * 1000.0),
            "ann_queries_per_sec": float(n / max(search_elapsed, 1e-9)),
        },
    }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
