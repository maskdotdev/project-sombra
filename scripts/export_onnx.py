from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from train_v1 import ContrastiveEmbedder, TrainConfig, select_device


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: ContrastiveEmbedder) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.encode(input_ids, attention_mask)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export embedding checkpoint to ONNX and optionally benchmark it.")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint-*/training_state.pt")
    p.add_argument("--output", default="artifacts/model-v1.onnx")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device-index", type=int, default=None)
    p.add_argument("--quantize-int8", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--bench-runs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    return p.parse_args()


def export_onnx(args: argparse.Namespace) -> tuple[Path, dict]:
    state = torch.load(args.checkpoint, map_location="cpu")
    cfg = TrainConfig(**state["config"])
    tokenizer_path = args.tokenizer_path or str(Path(args.checkpoint).parent / "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    device = select_device(args.device_index)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = ContrastiveEmbedder(cfg, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(state["model"], strict=True)
    model.eval().to(device)
    wrapper = OnnxExportWrapper(model).eval().to(device)

    dummy_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.seq_len), device=device, dtype=torch.long)
    dummy_mask = torch.ones((args.batch_size, args.seq_len), device=device, dtype=torch.long)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(out),
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embeddings": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    quantized_path = None
    if args.quantize_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = out.with_name(out.stem + ".int8.onnx")
        quantize_dynamic(str(out), str(quantized_path), weight_type=QuantType.QInt8)

    meta = {
        "checkpoint": args.checkpoint,
        "tokenizer_path": tokenizer_path,
        "onnx_path": str(out),
        "quantized_path": str(quantized_path) if quantized_path else None,
        "config": state["config"],
    }
    with (out.parent / "export_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return (quantized_path or out), meta


def benchmark_onnx(onnx_path: Path, args: argparse.Namespace) -> None:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"] if args.cpu_only else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    ids = np.random.randint(0, 1000, size=(args.batch_size, args.seq_len), dtype=np.int64)
    mask = np.ones((args.batch_size, args.seq_len), dtype=np.int64)
    inputs = {"input_ids": ids, "attention_mask": mask}

    for _ in range(10):
        sess.run(None, inputs)

    t0 = time.perf_counter()
    for _ in range(args.bench_runs):
        sess.run(None, inputs)
    elapsed = time.perf_counter() - t0
    ms = (elapsed / args.bench_runs) * 1000
    qps = args.batch_size / (elapsed / args.bench_runs)
    print(f"onnx_latency_ms={ms:.3f} batch_size={args.batch_size}")
    print(f"onnx_samples_per_sec={qps:,.1f}")
    print(f"providers={sess.get_providers()}")


def main() -> int:
    args = parse_args()
    final_path, _ = export_onnx(args)
    print(f"exported ONNX to {final_path}")
    if args.benchmark:
        benchmark_onnx(final_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
