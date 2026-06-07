from __future__ import annotations

import ast
import warnings
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from .utils import data_path, stable_hash, write_jsonl


def _synthetic_examples() -> list[dict[str, str]]:
    templates = [
        ("sum_positive", "Return the sum of positive numbers.", "def sum_positive(xs):\n    total = 0\n    for x in xs:\n        if x > 0:\n            total += x\n    return total\n"),
        ("normalize_name", "Normalize a user name string.", "def normalize_name(name):\n    clean = name.strip().lower()\n    return clean.replace(' ', '_')\n"),
        ("safe_divide", "Divide two numbers with a default for zero.", "def safe_divide(a, b, default=0):\n    if b == 0:\n        return default\n    return a / b\n"),
        ("count_words", "Count words in a string.", "def count_words(text):\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n"),
        ("flatten", "Flatten a list of lists.", "def flatten(items):\n    out = []\n    for row in items:\n        for value in row:\n            out.append(value)\n    return out\n"),
        ("clip", "Clip a value into a numeric interval.", "def clip(value, low, high):\n    if value < low:\n        return low\n    if value > high:\n        return high\n    return value\n"),
        ("parse_ints", "Parse integers from comma separated text.", "def parse_ints(text):\n    result = []\n    for part in text.split(','):\n        if part.strip():\n            result.append(int(part))\n    return result\n"),
        ("is_palindrome", "Check whether text is a palindrome.", "def is_palindrome(text):\n    s = ''.join(ch.lower() for ch in text if ch.isalnum())\n    return s == s[::-1]\n"),
        ("merge_dicts", "Merge dictionaries by summing values.", "def merge_dicts(left, right):\n    merged = dict(left)\n    for key, value in right.items():\n        merged[key] = merged.get(key, 0) + value\n    return merged\n"),
        ("first_match", "Return the first item satisfying a predicate.", "def first_match(items, pred):\n    for item in items:\n        if pred(item):\n            return item\n    return None\n"),
    ]
    rows = []
    for rep in range(3):
        for name, summary, code in templates:
            code2 = code.replace(name, f"{name}_{rep}")
            rows.append({"name": f"{name}_{rep}", "summary": summary, "code": code2})
    return rows


def _valid_function(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.body for node in tree.body)


def _first_nonempty(ex: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = ex.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return " ".join(str(v) for v in value if str(v).strip()).strip()
    return ""


def _iter_local_parquet(cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_dir = Path(cfg["data"]["local_parquet_dir"]).expanduser()
    files = sorted(str(p) for p in parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    for file in files:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=1024):
            for ex in batch.to_pylist():
                code = _first_nonempty(ex, ("func_code", "code", "function", "content"))
                summary = _first_nonempty(ex, ("func_documentation_string", "docstring", "summary", "description", "func_documentation_tokens"))
                name = _first_nonempty(ex, ("func_name", "function_name", "name"))
                source_id = _first_nonempty(ex, ("repo", "repository", "path", "url"))
                yield {"name": name, "summary": summary, "code": code, "source_id": source_id or "local_parquet"}


def prepare_dataset(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if cfg.get("mode") == "smoke" or cfg.get("data", {}).get("fallback_synthetic"):
        print("Using fallback smoke dataset; not suitable for final results.")
        raw: Iterable[dict[str, Any]] = _synthetic_examples()
    elif cfg.get("data", {}).get("local_parquet_dir"):
        raw = _iter_local_parquet(cfg)
    else:
        try:
            from datasets import load_dataset

            ds = load_dataset(cfg["data"].get("hf_dataset_name", "code_search_net"), cfg["data"].get("hf_dataset_config", "python"))
            hf_rows = []
            for split in ("train", "validation", "test"):
                for ex in ds[split]:
                    code = _first_nonempty(ex, ("func_code", "code"))
                    summary = _first_nonempty(ex, ("func_documentation_tokens", "func_documentation_string", "docstring", "summary"))
                    hf_rows.append({"name": ex.get("func_name", ""), "summary": summary, "code": code, "source_id": ex.get("repo", "")})
            raw = hf_rows
        except Exception as exc:
            raise RuntimeError("CodeSearchNet could not be loaded. Use smoke.yaml for fallback synthetic data.") from exc

    limits = cfg["data"]
    split_sizes = {
        "train": int(limits["max_functions_train"]),
        "val": int(limits["max_functions_val"]),
        "test": int(limits["max_functions_test"]),
    }
    seen_hashes: set[str] = set()
    rows: list[dict[str, Any]] = []
    split_counts = {k: 0 for k in split_sizes}
    split_order = ["train", "val", "test"]
    split_idx = 0
    for idx, ex in enumerate(raw):
        code = ex["code"].strip() + "\n"
        if not _valid_function(code):
            continue
        code_hash = stable_hash(code)
        if limits.get("deduplicate_by_code_hash", True) and code_hash in seen_hashes:
            continue
        seen_hashes.add(code_hash)
        while split_idx < len(split_order) and split_counts[split_order[split_idx]] >= split_sizes[split_order[split_idx]]:
            split_idx += 1
        if split_idx >= len(split_order):
            break
        split = split_order[split_idx]
        split_counts[split] += 1
        rows.append({
            "function_id": f"{split}_{idx}_{code_hash[:8]}",
            "repo": ex.get("source_id", "synthetic"),
            "source_id": ex.get("source_id", "synthetic"),
            "code_hash": code_hash,
            "code": code,
            "summary": ex.get("summary", "") or f"Python function {ex.get('name', '')}.",
            "split": split,
        })
    path = data_path(cfg, "prepared_functions.jsonl")
    write_jsonl(path, rows)
    return rows
