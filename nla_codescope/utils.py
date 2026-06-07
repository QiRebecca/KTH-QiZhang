from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = _expand_env(yaml.safe_load(f))
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    root = repo_root()
    for key in ("output_dir", "data_dir"):
        Path(root, cfg.get(key, key)).mkdir(parents=True, exist_ok=True)
    Path(root, "figures").mkdir(exist_ok=True)
    Path(root, "docs").mkdir(exist_ok=True)


def out_path(cfg: dict[str, Any], name: str) -> Path:
    return repo_root() / cfg.get("output_dir", "artifacts") / name


def data_path(cfg: dict[str, Any], name: str) -> Path:
    return repo_root() / cfg.get("data_dir", "data") / name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_int(text: str) -> int:
    return int(stable_hash(text)[:16], 16) % (2**32)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_vectors(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    meta = read_jsonl(out_path(cfg, "activations.jsonl"))
    arr = np.load(out_path(cfg, "activations.npz"))["h"].astype(np.float32)
    return meta, arr


def save_vectors(cfg: dict[str, Any], meta: list[dict[str, Any]], h: np.ndarray) -> None:
    write_jsonl(out_path(cfg, "activations.jsonl"), meta)
    dtype = np.float16 if cfg.get("activation", {}).get("vector_dtype_store") == "float16" else np.float32
    np.savez_compressed(out_path(cfg, "activations.npz"), h=h.astype(dtype))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def short_token_count(text: str) -> int:
    return len(text.split())
