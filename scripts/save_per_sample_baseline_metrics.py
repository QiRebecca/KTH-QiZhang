from __future__ import annotations

"""Save per-sample directional metrics for main methods and controls.

This script avoids retraining. It reuses saved roundtrip predictions/texts when
available and only reruns cheap inference for controls whose per-row predictions
were not previously saved.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from common import ar_artifact_path, av_artifact_path, load_bootstrap_map, parse_config, split_indices, subset
from nla_codescope.ar import load_ar
from nla_codescope.av import load_av
from nla_codescope.baselines import role_preserving_shuffle
from nla_codescope.bootstrap_texts import make_bootstrap_text
from nla_codescope.metrics import compute_metrics
from nla_codescope.utils import l2_normalize, out_path, read_jsonl, read_vectors, write_json


METHODS = [
    "Mean predictor",
    "Shuffled AV text",
    "Role-preserving shuffled AV text",
    "No-injection AV",
    "Deterministic template/bootstrap text -> AR_eval",
    "AV-SFT -> AR_eval",
    "AV-RerankSFT -> AR_eval",
]


def _load_roundtrip_cache(cfg: dict[str, Any], ids: list[str]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    text_path = out_path(cfg, "roundtrip_outputs.jsonl")
    pred_path = out_path(cfg, "roundtrip_predictions.npz")
    if not text_path.exists() or not pred_path.exists():
        raise FileNotFoundError("roundtrip_outputs.jsonl and roundtrip_predictions.npz are required")
    rows = read_jsonl(text_path)
    if [r.get("activation_id") for r in rows] != ids:
        raise ValueError("roundtrip_outputs.jsonl activation ids do not match test split")
    pred = np.load(pred_path)
    pred_ids = [str(x) for x in pred["activation_ids"].tolist()]
    if pred_ids != ids:
        raise ValueError("roundtrip_predictions.npz activation ids do not match test split")
    return (
        [str(r["av_sft_explanation"]) for r in rows],
        [str(r["av_rerank_explanation"]) for r in rows],
        pred["pred_sft"].astype(np.float32),
        pred["pred_rerank"].astype(np.float32),
    )


def _components(h: np.ndarray, pred: np.ndarray, h_train: np.ndarray) -> dict[str, np.ndarray]:
    h64 = h.astype(np.float64)
    p64 = pred.astype(np.float64)
    train64 = h_train.astype(np.float64)
    mean_train = train64.mean(axis=0, keepdims=True)
    h_norm = l2_normalize(h64)
    p_norm = l2_normalize(p64)
    mean_train_norm = l2_normalize(train64).mean(axis=0, keepdims=True)
    cosine = np.sum(h_norm * p_norm, axis=1)
    mse_nrm = np.sum((h_norm - p_norm) ** 2, axis=1)
    return {
        "cosine": cosine.astype(np.float64),
        "MSE_nrm": mse_nrm.astype(np.float64),
        "dir_num": mse_nrm.astype(np.float64),
        "dir_den": np.sum((h_norm - mean_train_norm) ** 2, axis=1).astype(np.float64),
        "raw_num": np.sum((h64 - p64) ** 2, axis=1).astype(np.float64),
        "raw_den": np.sum((h64 - mean_train) ** 2, axis=1).astype(np.float64),
    }


def main() -> None:
    cfg = parse_config()
    meta, h = read_vectors(cfg)
    train_idx = split_indices(meta, "train")
    test_idx = split_indices(meta, "test")
    h_train = subset(h, train_idx)
    h_test = subset(h, test_idx)
    rows_test = [meta[i] for i in test_idx]
    ids = [r["activation_id"] for r in rows_test]
    roles = [r["token_role"] for r in rows_test]
    function_ids = [r["function_id"] for r in rows_test]

    sft_texts, rerank_texts, pred_sft, pred_rerank = _load_roundtrip_cache(cfg, ids)
    ar_eval = load_ar(cfg, ar_artifact_path(cfg, "ar_eval"))

    rng = np.random.default_rng(int(cfg.get("seed", 17)))
    shuffled_texts = list(rerank_texts)
    rng.shuffle(shuffled_texts)
    role_shuffled_texts = role_preserving_shuffle(rerank_texts, roles, int(cfg.get("seed", 17)) + 101)
    bootstrap = load_bootstrap_map(cfg)
    bootstrap_texts = [bootstrap[aid] for aid in ids]
    context_texts = [make_bootstrap_text(r, int(cfg["bootstrap_text"]["max_tokens"])) for r in rows_test]
    if context_texts != bootstrap_texts:
        raise ValueError("Context-only and bootstrap texts differ; this script expects the merged deterministic template baseline.")

    av_rerank = load_av(cfg, av_artifact_path(cfg, "av_rerank"))
    no_injection_texts = av_rerank.generate(h_test, ids, mode="no_injection_cached", no_injection=True)

    pred_by_method = {
        "Mean predictor": np.repeat(h_train.mean(axis=0, keepdims=True), h_test.shape[0], axis=0).astype(np.float32),
        "Shuffled AV text": ar_eval.predict(shuffled_texts),
        "Role-preserving shuffled AV text": ar_eval.predict(role_shuffled_texts),
        "No-injection AV": ar_eval.predict(no_injection_texts),
        "Deterministic template/bootstrap text -> AR_eval": ar_eval.predict(bootstrap_texts),
        "AV-SFT -> AR_eval": pred_sft,
        "AV-RerankSFT -> AR_eval": pred_rerank,
    }

    method_keys = np.array(METHODS)
    arrays: dict[str, Any] = {
        "activation_ids": np.array(ids),
        "function_ids": np.array(function_ids),
        "token_roles": np.array(roles),
        "method_keys": method_keys,
    }
    aggregate: dict[str, Any] = {}
    for metric_name in ("cosine", "MSE_nrm", "dir_num", "dir_den", "raw_num", "raw_den"):
        arrays[metric_name] = np.stack([_components(h_test, pred_by_method[m], h_train)[metric_name] for m in METHODS])
    for method in METHODS:
        aggregate[method] = compute_metrics(h_test, pred_by_method[method], h_train)
    metrics_path = out_path(cfg, "per_sample_metrics_main.npz")
    np.savez_compressed(metrics_path, **arrays)
    write_json(out_path(cfg, "per_sample_metrics_main_summary.json"), aggregate)
    try:
        display = metrics_path.relative_to(Path.cwd())
    except ValueError:
        display = metrics_path
    print(f"wrote {display}")


if __name__ == "__main__":
    main()
