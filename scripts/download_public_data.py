from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
        return " ".join(value).strip()
    return str(value).strip()


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


@dataclass
class PullResult:
    source: str
    split: str
    status: str
    count: int
    error: str | None = None


def pull_code_x_glue(train_out: Path, eval_out: Path, per_lang_train: int, per_lang_eval: int) -> list[PullResult]:
    results: list[PullResult] = []
    langs = ["python", "javascript", "go", "java", "ruby", "php"]
    for lang in langs:
        source = f"google/code_x_glue_ct_code_to_text:{lang}"
        try:
            train_rows = []
            ds_train = load_dataset("google/code_x_glue_ct_code_to_text", lang, split="train", streaming=True)
            for ex in ds_train:
                anchor = _to_text(ex.get("docstring"))
                positive = _to_text(ex.get("code"))
                if anchor and positive:
                    train_rows.append({"anchor": anchor, "positive": positive, "source": source, "split": "train"})
                if len(train_rows) >= per_lang_train:
                    break
            train_count = _append_jsonl(train_out, train_rows)

            eval_rows = []
            ds_eval = load_dataset("google/code_x_glue_ct_code_to_text", lang, split="validation", streaming=True)
            for ex in ds_eval:
                anchor = _to_text(ex.get("docstring"))
                positive = _to_text(ex.get("code"))
                if anchor and positive:
                    eval_rows.append(
                        {"anchor": anchor, "positive": positive, "source": source, "split": "validation"}
                    )
                if len(eval_rows) >= per_lang_eval:
                    break
            eval_count = _append_jsonl(eval_out, eval_rows)

            results.append(PullResult(source=source, split="train", status="ok", count=train_count))
            results.append(PullResult(source=source, split="validation", status="ok", count=eval_count))
        except Exception as exc:
            results.append(PullResult(source=source, split="both", status="error", count=0, error=str(exc)))
    return results


def pull_mbpp(train_out: Path, eval_out: Path, train_limit: int, eval_limit: int) -> list[PullResult]:
    try:
        ds_train = load_dataset("mbpp", "full", split="train")
        ds_eval = load_dataset("mbpp", "full", split="validation")
        train_rows = []
        for ex in ds_train:
            anchor = _to_text(ex.get("text"))
            positive = _to_text(ex.get("code"))
            if anchor and positive:
                train_rows.append({"anchor": anchor, "positive": positive, "source": "mbpp", "split": "train"})
            if len(train_rows) >= train_limit:
                break

        eval_rows = []
        for ex in ds_eval:
            anchor = _to_text(ex.get("text"))
            positive = _to_text(ex.get("code"))
            if anchor and positive:
                eval_rows.append({"anchor": anchor, "positive": positive, "source": "mbpp", "split": "validation"})
            if len(eval_rows) >= eval_limit:
                break

        return [
            PullResult(source="mbpp", split="train", status="ok", count=_append_jsonl(train_out, train_rows)),
            PullResult(source="mbpp", split="validation", status="ok", count=_append_jsonl(eval_out, eval_rows)),
        ]
    except Exception as exc:
        return [PullResult(source="mbpp", split="both", status="error", count=0, error=str(exc))]


def pull_humaneval(eval_out: Path, eval_limit: int) -> list[PullResult]:
    try:
        ds = load_dataset("openai_humaneval", split="test")
        rows = []
        for ex in ds:
            prompt = _to_text(ex.get("prompt"))
            solution = _to_text(ex.get("canonical_solution"))
            if prompt and solution:
                rows.append(
                    {
                        "anchor": prompt,
                        "positive": prompt + "\n" + solution,
                        "source": "openai_humaneval",
                        "split": "test",
                    }
                )
            if len(rows) >= eval_limit:
                break
        return [PullResult(source="openai_humaneval", split="test", status="ok", count=_append_jsonl(eval_out, rows))]
    except Exception as exc:
        return [PullResult(source="openai_humaneval", split="test", status="error", count=0, error=str(exc))]


def pull_leetcode_pairs(train_out: Path, eval_out: Path, train_limit: int, eval_limit: int) -> list[PullResult]:
    try:
        ds = load_dataset("greengerong/leetcode", split="train", streaming=True)
        train_rows = []
        eval_rows = []
        for ex in ds:
            prompt = _to_text(ex.get("content"))
            code = _to_text(ex.get("python"))
            if prompt and code:
                row = {
                    "anchor": prompt,
                    "positive": code,
                    "source": "greengerong/leetcode",
                    "split": "train",
                    "difficulty": _to_text(ex.get("difficulty")),
                }
                if len(train_rows) < train_limit:
                    train_rows.append(row)
                elif len(eval_rows) < eval_limit:
                    row["split"] = "validation"
                    eval_rows.append(row)
            if len(train_rows) >= train_limit and len(eval_rows) >= eval_limit:
                break
        return [
            PullResult(
                source="greengerong/leetcode", split="train", status="ok", count=_append_jsonl(train_out, train_rows)
            ),
            PullResult(
                source="greengerong/leetcode",
                split="validation",
                status="ok",
                count=_append_jsonl(eval_out, eval_rows),
            ),
        ]
    except Exception as exc:
        return [PullResult(source="greengerong/leetcode", split="both", status="error", count=0, error=str(exc))]


def pull_raw_code(raw_out: Path, raw_limit: int) -> list[PullResult]:
    sources = [
        ("code-rag-bench/programming-solutions", "text"),
        ("code-rag-bench/stackoverflow-posts", "text"),
        ("code-rag-bench/online-tutorials", "text"),
    ]
    results: list[PullResult] = []
    remaining = raw_limit
    for name, text_key in sources:
        if remaining <= 0:
            break
        take = max(1, remaining // (len(sources) - len(results)))
        try:
            ds = load_dataset(name, split="train", streaming=True)
            rows = []
            for ex in ds:
                text = _to_text(ex.get(text_key))
                if text:
                    rows.append({"text": text, "source": name, "split": "train"})
                if len(rows) >= take:
                    break
            count = _append_jsonl(raw_out, rows)
            remaining -= count
            results.append(PullResult(source=name, split="train", status="ok", count=count))
        except Exception as exc:
            results.append(PullResult(source=name, split="train", status="error", count=0, error=str(exc)))
    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Download public code datasets and normalize to JSONL.")
    p.add_argument("--out-dir", default="data/public")
    p.add_argument("--xglue-train-per-lang", type=int, default=3000)
    p.add_argument("--xglue-eval-per-lang", type=int, default=500)
    p.add_argument("--mbpp-train", type=int, default=374)
    p.add_argument("--mbpp-eval", type=int, default=90)
    p.add_argument("--humaneval-eval", type=int, default=164)
    p.add_argument("--leetcode-train", type=int, default=3000)
    p.add_argument("--leetcode-eval", type=int, default=500)
    p.add_argument("--raw-code", type=int, default=12000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pairs = out_dir / "train_pairs.jsonl"
    eval_pairs = out_dir / "eval_pairs.jsonl"
    raw_text = out_dir / "raw_code_text.jsonl"
    for path in [train_pairs, eval_pairs, raw_text]:
        if path.exists():
            path.unlink()

    results: list[PullResult] = []
    results.extend(pull_code_x_glue(train_pairs, eval_pairs, args.xglue_train_per_lang, args.xglue_eval_per_lang))
    results.extend(pull_leetcode_pairs(train_pairs, eval_pairs, args.leetcode_train, args.leetcode_eval))
    results.extend(pull_mbpp(train_pairs, eval_pairs, args.mbpp_train, args.mbpp_eval))
    results.extend(pull_humaneval(eval_pairs, args.humaneval_eval))
    results.extend(pull_raw_code(raw_text, args.raw_code))

    ok = [r for r in results if r.status == "ok"]
    err = [r for r in results if r.status != "ok"]
    summary = {
        "output_dir": str(out_dir),
        "files": {
            "train_pairs": str(train_pairs),
            "eval_pairs": str(eval_pairs),
            "raw_code_text": str(raw_text),
        },
        "results": [asdict(r) for r in results],
        "totals": {
            "ok_sources": len(ok),
            "error_sources": len(err),
            "train_pairs": _line_count(train_pairs),
            "eval_pairs": _line_count(eval_pairs),
            "raw_text": _line_count(raw_text),
        },
    }

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["totals"], indent=2))
    if err:
        print("Some sources failed; see manifest.json for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
