from __future__ import annotations

import numpy as np
import pytest

from nla_codescope.injection import assert_activation_only_batch, build_activation_only_batch


def test_final_av_eval_batch_is_activation_only() -> None:
    batch = build_activation_only_batch("fixed <ACT> prompt", np.zeros((2, 4), dtype=np.float32), ["a", "b"])
    assert_activation_only_batch(batch)
    assert set(batch[0]) == {"activation_id", "prompt", "activation_vector"}


def test_context_field_is_rejected() -> None:
    batch = [{"activation_id": "x", "prompt": "p", "activation_vector": np.zeros(4), "summary_text": "bad"}]
    with pytest.raises(AssertionError):
        assert_activation_only_batch(batch)
