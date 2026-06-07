from __future__ import annotations

from typing import Any

import numpy as np

from .utils import l2_normalize


def cosine_mean(h: np.ndarray, h_hat: np.ndarray) -> float:
    return float(np.mean(np.sum(l2_normalize(h) * l2_normalize(h_hat), axis=1)))


def compute_metrics(h: np.ndarray, h_hat: np.ndarray, train_h: np.ndarray) -> dict[str, float]:
    h = h.astype(np.float64)
    h_hat = h_hat.astype(np.float64)
    train_h = train_h.astype(np.float64)
    if h.shape != h_hat.shape:
        raise ValueError(f"h and h_hat shape mismatch: {h.shape} vs {h_hat.shape}")
    mean_train = train_h.mean(axis=0, keepdims=True)
    raw_num = np.sum((h - h_hat) ** 2)
    raw_den = np.sum((h - mean_train) ** 2)
    train_norm = l2_normalize(train_h)
    h_norm = l2_normalize(h)
    h_hat_norm = l2_normalize(h_hat)
    mean_train_norm = train_norm.mean(axis=0, keepdims=True)
    dir_num = np.sum((h_norm - h_hat_norm) ** 2)
    dir_den = np.sum((h_norm - mean_train_norm) ** 2)
    return {
        "FVE_raw": float(1.0 - raw_num / max(raw_den, 1e-12)),
        "FVE_dir": float(1.0 - dir_num / max(dir_den, 1e-12)),
        "cosine": cosine_mean(h, h_hat),
        "MSE_nrm": float(np.mean(np.sum((h_norm - h_hat_norm) ** 2, axis=1))),
    }


def per_row_scores(h: np.ndarray, h_hat: np.ndarray) -> dict[str, np.ndarray]:
    h_norm = l2_normalize(h)
    h_hat_norm = l2_normalize(h_hat)
    return {
        "cosine": np.sum(h_norm * h_hat_norm, axis=1),
        "MSE_nrm": np.sum((h_norm - h_hat_norm) ** 2, axis=1),
    }


def bootstrap_ci_by_function(
    rows: list[dict[str, Any]],
    h: np.ndarray,
    h_hat: np.ndarray,
    train_h: np.ndarray,
    samples: int = 1000,
    seed: int = 17,
) -> dict[str, list[float]]:
    by_fn: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_fn.setdefault(row["function_id"], []).append(i)
    fns = sorted(by_fn)
    if not fns:
        return {}
    h = h.astype(np.float64)
    h_hat = h_hat.astype(np.float64)
    train_h = train_h.astype(np.float64)
    if h.shape != h_hat.shape:
        raise ValueError(f"h and h_hat shape mismatch: {h.shape} vs {h_hat.shape}")

    mean_train = train_h.mean(axis=0, keepdims=True)
    train_norm = l2_normalize(train_h)
    mean_train_norm = train_norm.mean(axis=0, keepdims=True)
    h_norm = l2_normalize(h)
    h_hat_norm = l2_normalize(h_hat)

    raw_num_i = np.sum((h - h_hat) ** 2, axis=1)
    raw_den_i = np.sum((h - mean_train) ** 2, axis=1)
    dir_num_i = np.sum((h_norm - h_hat_norm) ** 2, axis=1)
    dir_den_i = np.sum((h_norm - mean_train_norm) ** 2, axis=1)
    cosine_i = np.sum(h_norm * h_hat_norm, axis=1)

    fn_raw_num = []
    fn_raw_den = []
    fn_dir_num = []
    fn_dir_den = []
    fn_cosine_sum = []
    fn_count = []
    for fn in fns:
        idx = np.array(by_fn[fn], dtype=np.int64)
        fn_raw_num.append(float(raw_num_i[idx].sum()))
        fn_raw_den.append(float(raw_den_i[idx].sum()))
        fn_dir_num.append(float(dir_num_i[idx].sum()))
        fn_dir_den.append(float(dir_den_i[idx].sum()))
        fn_cosine_sum.append(float(cosine_i[idx].sum()))
        fn_count.append(int(len(idx)))
    fn_raw_num_a = np.array(fn_raw_num, dtype=np.float64)
    fn_raw_den_a = np.array(fn_raw_den, dtype=np.float64)
    fn_dir_num_a = np.array(fn_dir_num, dtype=np.float64)
    fn_dir_den_a = np.array(fn_dir_den, dtype=np.float64)
    fn_cosine_sum_a = np.array(fn_cosine_sum, dtype=np.float64)
    fn_count_a = np.array(fn_count, dtype=np.float64)

    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {"FVE_raw": [], "FVE_dir": [], "cosine": [], "MSE_nrm": []}
    n_fn = len(fns)
    for _ in range(samples):
        sampled = rng.integers(0, n_fn, size=n_fn)
        raw_num = fn_raw_num_a[sampled].sum()
        raw_den = fn_raw_den_a[sampled].sum()
        dir_num = fn_dir_num_a[sampled].sum()
        dir_den = fn_dir_den_a[sampled].sum()
        count = fn_count_a[sampled].sum()
        vals["FVE_raw"].append(float(1.0 - raw_num / max(raw_den, 1e-12)))
        vals["FVE_dir"].append(float(1.0 - dir_num / max(dir_den, 1e-12)))
        vals["cosine"].append(float(fn_cosine_sum_a[sampled].sum() / max(count, 1.0)))
        vals["MSE_nrm"].append(float(dir_num / max(count, 1.0)))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}
