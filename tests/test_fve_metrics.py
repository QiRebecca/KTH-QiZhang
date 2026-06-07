from __future__ import annotations

import numpy as np

from nla_codescope.metrics import compute_metrics


def test_perfect_prediction_gives_fve_one() -> None:
    train = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    h = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    m = compute_metrics(h, h.copy(), train)
    assert m["FVE_raw"] > 0.999999
    assert m["FVE_dir"] > 0.999999
    assert m["cosine"] > 0.999999
    assert m["MSE_nrm"] < 1e-10


def test_mean_predictor_gives_raw_fve_zero() -> None:
    train = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    h = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pred = np.repeat(train.mean(axis=0, keepdims=True), h.shape[0], axis=0)
    m = compute_metrics(h, pred, train)
    assert abs(m["FVE_raw"]) < 1e-6


def test_random_prediction_is_low_or_negative() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(20, 8)).astype(np.float32)
    h = rng.normal(size=(10, 8)).astype(np.float32)
    pred = rng.normal(size=(10, 8)).astype(np.float32) * 3.0
    m = compute_metrics(h, pred, train)
    assert m["FVE_raw"] < 0.2


def test_directional_fve_ignores_scale_better_than_raw() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(size=(20, 6)).astype(np.float32)
    h = rng.normal(size=(10, 6)).astype(np.float32)
    pred = h * 5.0
    m = compute_metrics(h, pred, train)
    assert m["FVE_dir"] > 0.999999
    assert m["FVE_raw"] < m["FVE_dir"]
