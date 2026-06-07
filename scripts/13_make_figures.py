from __future__ import annotations

from pathlib import Path

from common import ar_artifact_path, av_artifact_path, parse_config, split_indices, subset
from nla_codescope.ar import load_ar
from nla_codescope.av import load_av
from nla_codescope.metrics import per_row_scores
from nla_codescope.plots import save_architecture, save_main_bar, save_reconstruction_distribution, save_token_role_breakdown
from nla_codescope.utils import ensure_dirs, out_path, read_vectors
import json
import numpy as np


def main() -> None:
    cfg = parse_config()
    ensure_dirs(cfg)
    default_figures = "figures_smoke" if cfg.get("mode") == "smoke" else "figures"
    fig_dir = Path(__file__).resolve().parents[1] / cfg.get("_figures_dir", default_figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_architecture(fig_dir / "architecture.png")
    metrics = json.load(open(out_path(cfg, "metrics_main.json"), "r", encoding="utf-8"))
    save_main_bar(fig_dir / "main_fve_bar.png", metrics)
    role_metrics = json.load(open(out_path(cfg, "metrics_by_token_role.json"), "r", encoding="utf-8"))
    save_token_role_breakdown(fig_dir / "token_role_breakdown.png", role_metrics)
    meta, h = read_vectors(cfg)
    test_idx = split_indices(meta, "test")
    h_test = subset(h, test_idx)
    ids = [meta[i]["activation_id"] for i in test_idx]
    pred_path = out_path(cfg, "roundtrip_predictions.npz")
    if pred_path.exists():
        pred_data = np.load(pred_path)
        saved_ids = [str(x) for x in pred_data["activation_ids"].tolist()]
        if saved_ids != ids:
            raise ValueError("roundtrip_predictions.npz activation_ids do not match current test split")
        sft_pred = pred_data["pred_sft"].astype(np.float32)
        rerank_pred = pred_data["pred_rerank"].astype(np.float32)
    else:
        ar_eval = load_ar(cfg, ar_artifact_path(cfg, "ar_eval"))
        av_sft = load_av(cfg, av_artifact_path(cfg, "av_sft"))
        av_rerank = load_av(cfg, av_artifact_path(cfg, "av_rerank"))
        sft_pred = ar_eval.predict(av_sft.generate(h_test, ids, mode="sft"))
        rerank_pred = ar_eval.predict(av_rerank.generate(h_test, ids, mode="rerank"))
    scores = {
        "AV-SFT": per_row_scores(h_test, sft_pred)["cosine"].tolist(),
        "AV-RerankSFT": per_row_scores(h_test, rerank_pred)["cosine"].tolist(),
    }
    save_reconstruction_distribution(fig_dir / "reconstruction_distribution.png", scores)
    print("wrote figures")


if __name__ == "__main__":
    main()
