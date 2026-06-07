from __future__ import annotations

from typing import Any

import numpy as np


FORBIDDEN_EVAL_FIELDS = {
    "code",
    "summary",
    "summary_text",
    "docstring",
    "prefix",
    "suffix",
    "token_text",
    "local_context_for_bootstrap_only",
    "context_excerpt_for_human_display_only",
    "function_name",
}


def compute_injection_scale(token_embedding_norms: np.ndarray, activations: np.ndarray) -> float:
    emb = float(np.median(token_embedding_norms))
    act = float(np.median(np.linalg.norm(activations, axis=1)))
    scale = emb / max(act, 1e-12)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Injection scale must be finite and positive")
    return scale


def build_activation_only_batch(prompt_template: str, activation_vectors: np.ndarray, activation_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    fixed_prompt = prompt_template
    for activation_id, vec in zip(activation_ids, activation_vectors):
        rows.append({"activation_id": activation_id, "prompt": fixed_prompt, "activation_vector": vec.astype(np.float32)})
    return rows


def assert_activation_only_batch(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        bad = FORBIDDEN_EVAL_FIELDS.intersection(row)
        if bad:
            raise AssertionError(f"Final AV evaluation batch contains forbidden context fields: {sorted(bad)}")
        if set(row) - {"activation_id", "prompt", "activation_vector"}:
            raise AssertionError(f"Unexpected AV eval fields: {sorted(set(row) - {'activation_id', 'prompt', 'activation_vector'})}")
