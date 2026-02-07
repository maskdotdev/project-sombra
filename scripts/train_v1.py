from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, BertConfig, BertModel, get_cosine_schedule_with_warmup


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested_index: int | None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if requested_index is not None:
        return torch.device(f"cuda:{requested_index}")
    best_i = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    return torch.device(f"cuda:{best_i}")


@dataclass
class TrainConfig:
    train_file: str
    output_dir: str
    tokenizer_path: str
    eval_file: str | None = None
    eval_every: int = 250
    eval_max_samples: int = 2000
    save_every: int = 500
    max_steps: int = 4000
    batch_size: int = 64
    grad_accum_steps: int = 1
    lr: float = 3e-4
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_seq_len: int = 256
    temperature: float = 0.05
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    embedding_dim: int = 384
    mlm_prob: float = 0.15
    hard_neg_weight: float = 0.3
    seed: int = 42
    num_workers: int = 4
    device_index: int | None = None
    compile: bool = False
    log_every: int = 20


class ContrastiveEmbedder(nn.Module):
    """BERT encoder + projection head for code embedding training from scratch."""

    def __init__(self, cfg: TrainConfig, vocab_size: int) -> None:
        super().__init__()
        model_cfg = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=cfg.max_seq_len + 2,
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertModel(model_cfg)
        self.proj = nn.Linear(cfg.hidden_size, cfg.embedding_dim, bias=False)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        emb = self.proj(pooled)
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        anchor_ids: torch.Tensor,
        anchor_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(anchor_ids, anchor_mask), self.encode(pos_ids, pos_mask)


def maybe_corrupt_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor, vocab_size: int, prob: float) -> torch.Tensor:
    if prob <= 0:
        return input_ids
    keep = torch.rand_like(input_ids, dtype=torch.float32) >= prob
    rand_ids = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)
    # Preserve pad positions to keep sequence lengths stable.
    keep = keep | (attention_mask == 0)
    return torch.where(keep, input_ids, rand_ids)


def info_nce_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float,
    neg: torch.Tensor | None = None,
    hard_neg_weight: float = 0.3,
) -> torch.Tensor:
    logits = a @ b.t()
    logits = logits / temperature
    labels = torch.arange(a.size(0), device=a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    base = 0.5 * (loss_ab + loss_ba)
    if neg is None:
        return base

    # Anchor->(positive + hard negative bank) adds explicit close-but-wrong negatives.
    extended = torch.cat([b, neg], dim=0)
    hard_logits = (a @ extended.t()) / temperature
    hard_loss = F.cross_entropy(hard_logits, labels)
    return (1.0 - hard_neg_weight) * base + hard_neg_weight * hard_loss


def load_pairs_dataset(path: str) -> Any:
    ds = load_dataset("json", data_files=path, split="train")
    cols = set(ds.column_names)
    if {"anchor", "positive"}.issubset(cols):
        return ds
    if "text" in cols:
        return ds.map(lambda x: {"anchor": x["text"], "positive": x["text"]}, remove_columns=list(cols))
    raise ValueError("Dataset must contain either columns ('anchor','positive') or ('text').")


def build_collate(tokenizer: Any, max_seq_len: int):
    def collate(batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        anchors = [x["anchor"] for x in batch]
        positives = [x["positive"] for x in batch]
        has_hard_neg = all(bool(x.get("negative")) for x in batch)
        a = tokenizer(
            anchors,
            truncation=True,
            max_length=max_seq_len,
            padding=True,
            return_tensors="pt",
        )
        p = tokenizer(
            positives,
            truncation=True,
            max_length=max_seq_len,
            padding=True,
            return_tensors="pt",
        )
        out = {
            "anchor_ids": a["input_ids"],
            "anchor_mask": a["attention_mask"],
            "pos_ids": p["input_ids"],
            "pos_mask": p["attention_mask"],
        }
        if has_hard_neg:
            negatives = [x["negative"] for x in batch]
            n = tokenizer(
                negatives,
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_tensors="pt",
            )
            out["neg_ids"] = n["input_ids"]
            out["neg_mask"] = n["attention_mask"]
        return out

    return collate


@torch.no_grad()
def evaluate_recall(
    model: ContrastiveEmbedder,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    model.eval()
    all_a: list[torch.Tensor] = []
    all_p: list[torch.Tensor] = []
    seen = 0
    for batch in eval_loader:
        anchor_ids = batch["anchor_ids"].to(device, non_blocking=True)
        anchor_mask = batch["anchor_mask"].to(device, non_blocking=True)
        pos_ids = batch["pos_ids"].to(device, non_blocking=True)
        pos_mask = batch["pos_mask"].to(device, non_blocking=True)
        a = model.encode(anchor_ids, anchor_mask)
        p = model.encode(pos_ids, pos_mask)
        all_a.append(a.cpu())
        all_p.append(p.cpu())
        seen += a.size(0)
        if seen >= max_samples:
            break

    a = torch.cat(all_a, dim=0)[:max_samples]
    p = torch.cat(all_p, dim=0)[:max_samples]
    sims = a @ p.t()
    ranks = sims.argsort(dim=1, descending=True)
    target = torch.arange(ranks.size(0)).unsqueeze(1)
    hit_1 = (ranks[:, :1] == target).any(dim=1).float().mean().item()
    hit_5 = (ranks[:, :5] == target).any(dim=1).float().mean().item()
    hit_10 = (ranks[:, :10] == target).any(dim=1).float().mean().item()
    model.train()
    return {"recall@1": hit_1, "recall@5": hit_5, "recall@10": hit_10}


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: TrainConfig,
    tokenizer: Any,
    metrics: dict[str, float] | None = None,
) -> None:
    out = ckpt_dir / f"checkpoint-{step}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics or {},
        },
        out / "training_state.pt",
    )
    tokenizer.save_pretrained(str(out / "tokenizer"))


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a code embedding model from scratch (v1 scaffold).")
    p.add_argument("--train-file", required=True)
    p.add_argument("--output-dir", default="runs/v1")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--eval-file")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-max-samples", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-hidden-layers", type=int, default=8)
    p.add_argument("--num-attention-heads", type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--embedding-dim", type=int, default=384)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--hard-neg-weight", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device-index", type=int)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()
    return TrainConfig(**vars(args))


def main() -> int:
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(Path(cfg.output_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = select_device(cfg.device_index)
    using_cuda = device.type == "cuda"
    if using_cuda:
        torch.cuda.set_device(device)
    bf16 = using_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    train_ds = load_pairs_dataset(cfg.train_file)
    eval_ds = load_pairs_dataset(cfg.eval_file) if cfg.eval_file else None

    collate_fn = build_collate(tokenizer, cfg.max_seq_len)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=using_cuda,
        collate_fn=collate_fn,
        drop_last=True,
    )
    eval_loader = None
    if eval_ds is not None:
        eval_loader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=using_cuda,
            collate_fn=collate_fn,
            drop_last=False,
        )

    model = ContrastiveEmbedder(cfg, vocab_size=tokenizer.vocab_size).to(device)
    if cfg.compile:
        model = cast(ContrastiveEmbedder, torch.compile(model))

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=using_cuda)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_opt_steps = cfg.max_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_opt_steps,
    )

    scaler = None
    if using_cuda and (not bf16):
        amp_mod = getattr(torch, "amp", None)
        if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
            scaler = amp_mod.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("=== v1 Training Start ===")
    print(f"device={device}")
    if using_cuda:
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"train_samples={len(train_ds)} eval_samples={len(eval_ds) if eval_ds is not None else 0}")

    step = 0
    epoch = 0
    best_r10 = -1.0
    t0 = time.perf_counter()
    running_loss = 0.0

    model.train()
    while step < cfg.max_steps:
        epoch += 1
        for batch in train_loader:
            step += 1
            anchor_ids = batch["anchor_ids"].to(device, non_blocking=True)
            anchor_mask = batch["anchor_mask"].to(device, non_blocking=True)
            pos_ids = batch["pos_ids"].to(device, non_blocking=True)
            pos_mask = batch["pos_mask"].to(device, non_blocking=True)
            neg_ids = batch["neg_ids"].to(device, non_blocking=True) if "neg_ids" in batch else None
            neg_mask = batch["neg_mask"].to(device, non_blocking=True) if "neg_mask" in batch else None

            pos_ids = maybe_corrupt_ids(pos_ids, pos_mask, tokenizer.vocab_size, cfg.mlm_prob)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=using_cuda):
                a, p = model(anchor_ids, anchor_mask, pos_ids, pos_mask)
                n = model.encode(neg_ids, neg_mask) if neg_ids is not None and neg_mask is not None else None
                loss = info_nce_loss(a, p, cfg.temperature, neg=n, hard_neg_weight=cfg.hard_neg_weight)
                loss = loss / cfg.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * cfg.grad_accum_steps

            if step % cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = running_loss / cfg.log_every
                running_loss = 0.0
                tok_per_step = cfg.batch_size * cfg.max_seq_len * 2
                tps = (tok_per_step * cfg.log_every) / max(elapsed, 1e-6)
                lr = scheduler.get_last_lr()[0]
                print(f"step={step} epoch={epoch} loss={avg_loss:.4f} lr={lr:.6f} tok/s={tps:,.0f}")
                t0 = time.perf_counter()

            if eval_loader is not None and step % cfg.eval_every == 0:
                metrics = evaluate_recall(model, eval_loader, device, cfg.eval_max_samples)
                print(
                    f"eval step={step} recall@1={metrics['recall@1']:.4f} "
                    f"recall@5={metrics['recall@5']:.4f} recall@10={metrics['recall@10']:.4f}"
                )
                if metrics["recall@10"] > best_r10:
                    best_r10 = metrics["recall@10"]
                    save_checkpoint(Path(cfg.output_dir), step, model, optimizer, scheduler, cfg, tokenizer, metrics)
                    print(f"saved best checkpoint at step={step}")

            if step % cfg.save_every == 0:
                save_checkpoint(Path(cfg.output_dir), step, model, optimizer, scheduler, cfg, tokenizer)
                print(f"saved checkpoint at step={step}")

            if step >= cfg.max_steps:
                break

    save_checkpoint(Path(cfg.output_dir), step, model, optimizer, scheduler, cfg, tokenizer)
    print(f"training complete: steps={step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
