from __future__ import annotations

import numpy as np

from nla_codescope.injection import compute_injection_scale


def test_injection_shape_and_scale() -> None:
    d_model = 64
    token_embedding = np.ones((10, d_model), dtype=np.float32)
    activations = np.ones((5, d_model), dtype=np.float32) * 2
    scale = compute_injection_scale(np.linalg.norm(token_embedding, axis=1), activations)
    replacement = scale * activations[0]
    assert replacement.shape == (d_model,)
    assert np.isfinite(scale)
    assert scale > 0


def test_act_token_config_value() -> None:
    act_token = "<ACT>"
    assert act_token.startswith("<")
    assert act_token.endswith(">")
