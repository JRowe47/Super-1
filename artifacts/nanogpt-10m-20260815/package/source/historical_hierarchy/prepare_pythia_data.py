"""Stream and tokenize calibration text with an exact model tokenizer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="EleutherAI/pythia-1b-deduped")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--train_tokens", type=int, default=120_000_000)
    parser.add_argument("--val_tokens", type=int, default=2_000_000)
    parser.add_argument("--batch_documents", type=int, default=32)
    parser.add_argument(
        "--token_dtype", choices=("uint16", "uint32"), default="uint16",
        help="storage dtype; uint32 is required for vocabularies above 65,536")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def atomic_json(payload, path):
    temporary = str(path) + ".tmp"
    Path(temporary).write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


def complete(path, tokens, dtype):
    return path.exists() and path.stat().st_size == tokens * dtype.itemsize


def main():
    args = arguments()
    if args.train_tokens < 1 or args.val_tokens < 1:
        raise ValueError("train and validation token targets must be positive")
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install datasets and transformers before preparing data") from error

    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    token_dtype = np.dtype(args.token_dtype)
    train_final = output / "train.bin"
    val_final = output / "val.bin"
    if (complete(train_final, args.train_tokens, token_dtype)
            and complete(val_final, args.val_tokens, token_dtype)):
        print(json.dumps({"event": "already_complete", "out": str(output)}), flush=True)
        return

    train_partial = output / "train.bin.partial"
    val_partial = output / "val.bin.partial"
    progress_path = output / "progress.json"
    progress = {
        "documents": 0,
        "train_tokens": (train_partial.stat().st_size // token_dtype.itemsize
                         if train_partial.exists() else 0),
        "val_tokens": (val_partial.stat().st_size // token_dtype.itemsize
                       if val_partial.exists() else 0),
    }
    if progress_path.exists():
        recorded = json.loads(progress_path.read_text())
        if (recorded.get("train_tokens") == progress["train_tokens"]
                and recorded.get("val_tokens") == progress["val_tokens"]):
            progress["documents"] = int(recorded.get("documents", 0))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if len(tokenizer) > np.iinfo(token_dtype).max + 1:
        raise ValueError(
            f"the tokenizer vocabulary does not fit {args.token_dtype}")
    stream = load_dataset(
        args.dataset, args.subset, split=args.split, streaming=True)
    if progress["documents"]:
        stream = stream.skip(progress["documents"])
    iterator = iter(stream)

    train_handle = open(train_partial, "ab")
    val_handle = open(val_partial, "ab")
    try:
        while (progress["train_tokens"] < args.train_tokens
               or progress["val_tokens"] < args.val_tokens):
            rows = []
            for _ in range(args.batch_documents):
                rows.append(next(iterator))
            texts = [row[args.text_column] for row in rows]
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            for token_ids in encoded:
                sequence = token_ids + [tokenizer.eos_token_id]
                cursor = 0
                if progress["val_tokens"] < args.val_tokens:
                    count = min(
                        len(sequence), args.val_tokens - progress["val_tokens"])
                    np.asarray(sequence[:count], dtype=token_dtype).tofile(val_handle)
                    progress["val_tokens"] += count
                    cursor += count
                if cursor < len(sequence) and progress["train_tokens"] < args.train_tokens:
                    count = min(
                        len(sequence) - cursor,
                        args.train_tokens - progress["train_tokens"])
                    np.asarray(
                        sequence[cursor:cursor + count], dtype=token_dtype
                    ).tofile(train_handle)
                    progress["train_tokens"] += count
            progress["documents"] += len(rows)
            train_handle.flush()
            val_handle.flush()
            atomic_json(progress, progress_path)
            if progress["documents"] % (args.batch_documents * 20) == 0:
                print(json.dumps({"event": "progress", **progress}), flush=True)
    finally:
        train_handle.close()
        val_handle.close()

    if progress["train_tokens"] != args.train_tokens or progress["val_tokens"] != args.val_tokens:
        raise RuntimeError(f"tokenization ended short: {progress}")
    os.replace(train_partial, train_final)
    os.replace(val_partial, val_final)
    progress_path.unlink(missing_ok=True)
    print(json.dumps({
        "event": "done", "train_tokens": args.train_tokens,
        "val_tokens": args.val_tokens, "token_dtype": args.token_dtype,
        "out": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()
