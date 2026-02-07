from __future__ import annotations

import argparse
import time
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyEmbedder(nn.Module):
    """Small encoder used to validate CUDA training throughput and stability."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_dim, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Positional indices are built on-device to avoid host->device copies each step.
        bsz, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.token(input_ids) + self.pos(pos_ids)
        x = self.encoder(x)
        x = self.norm(x)

        # Mean pooling with L2 normalization mirrors common embedding serving setups.
        pooled = x.mean(dim=1)
        return F.normalize(self.proj(pooled), p=2, dim=-1)


def select_device(requested_index: int | None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if requested_index is not None:
        return torch.device(f"cuda:{requested_index}")

    # Default to the largest VRAM GPU for training headroom.
    best_i = max(
        range(torch.cuda.device_count()),
        key=lambda i: torch.cuda.get_device_properties(i).total_memory,
    )
    return torch.device(f"cuda:{best_i}")


def make_positive_view(input_ids: torch.Tensor, vocab_size: int, noise_prob: float) -> torch.Tensor:
    # Create a lightly corrupted view so each sample has a semantically-close positive.
    if noise_prob <= 0:
        return input_ids
    mask = torch.rand_like(input_ids, dtype=torch.float32) < noise_prob
    random_tokens = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)
    return torch.where(mask, random_tokens, input_ids)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (z1 @ z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a fast mixed-precision GPU training sanity check.")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=50000)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--out-dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--noise-prob", type=float, default=0.15)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device-index", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    device = select_device(args.device_index)
    using_cuda = device.type == "cuda"
    bf16 = using_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16

    if using_cuda:
        torch.cuda.set_device(device)

    model = TinyEmbedder(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ff_mult=args.ff_mult,
        out_dim=args.out_dim,
        dropout=args.dropout,
    ).to(device)

    if args.compile:
        model = cast(nn.Module, torch.compile(model))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = None
    if using_cuda and (not bf16):
        amp_mod = getattr(torch, "amp", None)
        if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
            scaler = amp_mod.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("=== Training Sanity Check ===")
    print(f"device: {device}")
    if using_cuda:
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"amp_dtype: {amp_dtype}")
    print(f"steps: {args.steps}, warmup: {args.warmup}")
    print(f"batch_size: {args.batch_size}, seq_len: {args.seq_len}")

    total_tokens = 0
    measured_time = 0.0
    start = time.perf_counter()

    model.train()
    for step in range(1, args.steps + 1):
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        positive_ids = make_positive_view(input_ids, args.vocab_size, args.noise_prob)

        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=using_cuda):
            z1 = model(input_ids)
            z2 = model(positive_ids)
            loss = contrastive_loss(z1, z2, args.temperature)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        if using_cuda:
            torch.cuda.synchronize(device)

        if step > args.warmup:
            # Ignore warmup steps to report stable throughput.
            step_tokens = args.batch_size * args.seq_len * 2
            total_tokens += step_tokens

        now = time.perf_counter()
        step_time = now - start
        start = now
        if step > args.warmup:
            measured_time += step_time

        if step == 1 or step % 10 == 0 or step == args.steps:
            tok_s = (total_tokens / measured_time) if measured_time > 0 else 0.0
            print(f"step={step:4d} loss={loss.item():.4f} step_time={step_time:.3f}s tokens/s={tok_s:,.0f}")

    if using_cuda:
        mem = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"max_cuda_memory_allocated_gb: {mem:.2f}")

    if measured_time > 0:
        print(f"final_tokens_per_second: {total_tokens / measured_time:,.0f}")
    print("sanity_check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
