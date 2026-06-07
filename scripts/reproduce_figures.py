from __future__ import annotations

"""Regenerate figures from saved artifacts without model inference when possible."""

import json
from pathlib import Path

import numpy as np

from common import parse_config, split_indices, subset
from nla_codescope.metrics import per_row_scores
from nla_codescope.plots import save_architecture, save_main_bar, save_reconstruction_distribution, save_token_role_breakdown
from nla_codescope.utils import out_path, read_vectors


def main() -> None:
    cfg = parse_config()
    root = Path(__file__).resolve().parents[1]
    fig_dir = root / cfg.get("_figures_dir", "figures")
    fig_dir.mkdir(exist_ok=True)
    save_architecture(fig_dir / "architecture.png")
    save_main_bar(fig_dir / "main_fve_bar.png", json.load(open(out_path(cfg, "metrics_main.json"), "r", encoding="utf-8")))
    save_token_role_breakdown(
        fig_dir / "token_role_breakdown.png",
        json.load(open(out_path(cfg, "metrics_by_token_role.json"), "r", encoding="utf-8")),
    )

    pred_path = out_path(cfg, "roundtrip_predictions.npz")
    if pred_path.exists():
        meta, h = read_vectors(cfg)
        test_idx = split_indices(meta, "test")
        h_test = subset(h, test_idx)
        pred = np.load(pred_path)
        scores = {
            "AV-SFT": per_row_scores(h_test, pred["pred_sft"].astype(np.float32))["cosine"].tolist(),
            "AV-RerankSFT": per_row_scores(h_test, pred["pred_rerank"].astype(np.float32))["cosine"].tolist(),
        }
        save_reconstruction_distribution(fig_dir / "reconstruction_distribution.png", scores)
    else:
        print("Skipping reconstruction_distribution.png: roundtrip_predictions.npz is missing.")
    print("reproduced available figures from artifacts")


if __name__ == "__main__":
    main()
