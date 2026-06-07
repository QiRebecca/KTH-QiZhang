from __future__ import annotations

import numpy as np

from .metrics import per_row_scores


def choose_best_candidates(ar_train: object, h: np.ndarray, candidates: list[list[str]]) -> tuple[list[str], list[float]]:
    best_texts: list[str] = []
    best_scores: list[float] = []
    for target, texts in zip(h, candidates):
        preds = ar_train.predict(texts)
        scores = -per_row_scores(np.repeat(target[None, :], len(texts), axis=0), preds)["MSE_nrm"]
        idx = int(np.argmax(scores))
        best_texts.append(texts[idx])
        best_scores.append(float(scores[idx]))
    return best_texts, best_scores
