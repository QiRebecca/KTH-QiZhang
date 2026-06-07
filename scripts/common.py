from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nla_codescope.utils import load_config, out_path, read_jsonl


def parse_config() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_coder_15b_layer18.yaml")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--figures", default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.artifacts:
        cfg["output_dir"] = args.artifacts
    if args.figures:
        cfg["_figures_dir"] = args.figures
    cfg["_require_complete"] = bool(args.require_complete)
    return cfg


def split_indices(meta: list[dict[str, Any]], split: str) -> list[int]:
    return [i for i, row in enumerate(meta) if row["split"] == split]


def load_bootstrap_map(cfg: dict[str, Any]) -> dict[str, str]:
    return {r["activation_id"]: r["bootstrap_text"] for r in read_jsonl(out_path(cfg, "bootstrap_texts.jsonl"))}


def subset(arr: np.ndarray, idx: list[int]) -> np.ndarray:
    return arr[np.array(idx, dtype=np.int64)]


def ar_artifact_path(cfg: dict[str, Any], name: str) -> Path:
    if cfg["ar"].get("backend") == "hashed_ridge_smoke":
        return out_path(cfg, f"{name}.npz")
    return out_path(cfg, name)


def av_artifact_path(cfg: dict[str, Any], name: str) -> Path:
    if cfg["av"].get("backend") == "nearest_text_smoke":
        return out_path(cfg, f"{name}.npz")
    return out_path(cfg, name)
