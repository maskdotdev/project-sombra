from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from train_v1 import ContrastiveEmbedder, TrainConfig, select_device


@torch.no_grad()
def encode_texts(
    model: ContrastiveEmbedder,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_seq_len: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        toks = tokenizer(batch, truncation=True, max_length=max_seq_len, padding=True, return_tensors="pt")
        ids = toks["input_ids"].to(device, non_blocking=True)
        mask = toks["attention_mask"].to(device, non_blocking=True)
        emb = model.encode(ids, mask)
        chunks.append(emb.cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def mine_indices(anchor_emb: torch.Tensor, pos_emb: torch.Tensor, chunk_size: int) -> list[int]:
    n = anchor_emb.size(0)
    out: list[int] = []
    pos_t = pos_emb.t().contiguous()
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = anchor_emb[start:end] @ pos_t
        diag_rows = torch.arange(end - start)
        diag_cols = torch.arange(start, end)
        sims[diag_rows, diag_cols] = -1e9
        out.extend(sims.argmax(dim=1).tolist())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine hard negatives from a trained checkpoint.")
    p.add_argument("--input-file", required=True)
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint-*/training_state.pt")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--output-file", default="data/public/train_pairs_hard.jsonl")
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--device-index", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")
    cfg = TrainConfig(**state["config"])
    tokenizer_path = args.tokenizer_path or str(Path(args.checkpoint).parent / "tokenizer")

    device = select_device(args.device_index)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    model = ContrastiveEmbedder(cfg, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()

    ds = load_dataset("json", data_files=args.input_file, split="train")
    if not {"anchor", "positive"}.issubset(set(ds.column_names)):
        raise ValueError("Input file must contain 'anchor' and 'positive' columns.")

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    anchors = [str(x) for x in ds["anchor"]]
    positives = [str(x) for x in ds["positive"]]
    print(f"encoding samples={len(anchors)} on {device}")

    anchor_emb = encode_texts(model, tokenizer, anchors, device, args.batch_size, cfg.max_seq_len)
    pos_emb = encode_texts(model, tokenizer, positives, device, args.batch_size, cfg.max_seq_len)

    neg_idx = mine_indices(anchor_emb, pos_emb, args.chunk_size)
    out_rows = []
    for i, ex in enumerate(ds):
        row = {
            "anchor": ex["anchor"],
            "positive": ex["positive"],
            "negative": positives[neg_idx[i]],
            "source": ex.get("source", "unknown"),
            "split": ex.get("split", "train"),
        }
        out_rows.append(row)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"wrote {len(out_rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
