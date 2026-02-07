from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def iter_texts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if "text" in ex:
                yield ex["text"]
            else:
                if "anchor" in ex:
                    yield ex["anchor"]
                if "positive" in ex:
                    yield ex["positive"]


def main() -> int:
    p = argparse.ArgumentParser(description="Train a fast tokenizer for code embedding training.")
    p.add_argument("--train-file", required=True)
    p.add_argument("--output-dir", default="artifacts/tokenizer-v1")
    p.add_argument("--vocab-size", type=int, default=32000)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = Sequence([NFKC()])
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tok.train_from_iterator(iter_texts(args.train_file), trainer=trainer)
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    fast.save_pretrained(str(out))
    print(f"saved tokenizer to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
