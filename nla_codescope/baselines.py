from __future__ import annotations

import numpy as np

from .metrics import compute_metrics


def evaluate_text_method(texts: list[str], h: np.ndarray, train_h: np.ndarray, ar_eval: object) -> dict[str, float]:
    return compute_metrics(h, ar_eval.predict(texts), train_h)


def mean_predictor(h: np.ndarray, train_h: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    pred = np.repeat(train_h.mean(axis=0, keepdims=True), h.shape[0], axis=0)
    return pred, compute_metrics(h, pred, train_h)


def shuffled_text_metrics(texts: list[str], h: np.ndarray, train_h: np.ndarray, ar_eval: object, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)
    return evaluate_text_method(shuffled, h, train_h, ar_eval)


def role_preserving_shuffle(texts: list[str], roles: list[str], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    shuffled = list(texts)
    by_role: dict[str, list[int]] = {}
    for i, role in enumerate(roles):
        by_role.setdefault(role, []).append(i)
    for idxs in by_role.values():
        vals = [shuffled[i] for i in idxs]
        if len(vals) > 1:
            rng.shuffle(vals)
        for i, val in zip(idxs, vals):
            shuffled[i] = val
    return shuffled


def role_preserving_shuffled_text_metrics(
    texts: list[str],
    roles: list[str],
    h: np.ndarray,
    train_h: np.ndarray,
    ar_eval: object,
    seed: int,
) -> dict[str, float]:
    return evaluate_text_method(role_preserving_shuffle(texts, roles, seed), h, train_h, ar_eval)


def no_injection_metrics(av: object, h: np.ndarray, ids: list[str], train_h: np.ndarray, ar_eval: object) -> dict[str, float]:
    texts = av.generate(h, ids, no_injection=True)
    return evaluate_text_method(texts, h, train_h, ar_eval)
