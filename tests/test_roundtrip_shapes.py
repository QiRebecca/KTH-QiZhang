from __future__ import annotations

import numpy as np

from nla_codescope.ar import train_hashed_ridge
from nla_codescope.av import train_nearest_text_av
from nla_codescope.metrics import compute_metrics
from nla_codescope.utils import l2_normalize


def test_roundtrip_shapes_on_synthetic_data() -> None:
    rng = np.random.default_rng(0)
    h = rng.normal(size=(8, 16)).astype(np.float32)
    texts_train = [f"synthetic explanation {i} return value" for i in range(len(h))]
    cfg = {"ar": {"feature_dim": 32, "ridge_lambda": 0.1}}
    ar = train_hashed_ridge(texts_train, h, cfg)
    av = train_nearest_text_av(h, texts_train, {"av": {}})

    ids = [f"a{i}" for i in range(3)]
    texts = av.generate(h[: len(ids)], ids)
    assert all(isinstance(t, str) and t for t in texts)
    h_hat = ar.predict(texts)
    assert h_hat.shape == h[: len(ids)].shape
    assert ar.weights.shape[1] == h.shape[1]
    m = compute_metrics(h[: len(ids)], h_hat, h)
    assert set(m) == {"FVE_raw", "FVE_dir", "cosine", "MSE_nrm"}
    assert l2_normalize(h_hat).shape == h_hat.shape
