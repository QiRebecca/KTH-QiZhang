from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import ar_artifact_path, av_artifact_path, load_bootstrap_map, parse_config, split_indices, subset
from nla_codescope.ar import load_ar
from nla_codescope.av import load_av
from nla_codescope.baselines import (
    evaluate_text_method,
    mean_predictor,
    no_injection_metrics,
    role_preserving_shuffled_text_metrics,
    shuffled_text_metrics,
)
from nla_codescope.bootstrap_texts import make_bootstrap_text
from nla_codescope.metrics import bootstrap_ci_by_function, compute_metrics
from nla_codescope.utils import ensure_dirs, out_path, read_jsonl, read_vectors, write_json


def _load_existing_method_metrics(cfg: dict, method: str) -> dict[str, float] | None:
    path = out_path(cfg, "metrics_main.json")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        metrics = payload.get("methods", {}).get(method)
    except Exception:
        return None
    if isinstance(metrics, dict) and {"FVE_raw", "FVE_dir", "cosine", "MSE_nrm"}.issubset(metrics):
        return {k: float(metrics[k]) for k in ["FVE_raw", "FVE_dir", "cosine", "MSE_nrm"]}
    return None


def _cached_roundtrip(
    cfg: dict, ids: list[str], h_test, h_train
) -> tuple[list[str] | None, list[str] | None, dict[str, float] | None, dict[str, float] | None, np.ndarray | None]:
    pred_path = out_path(cfg, "roundtrip_predictions.npz")
    text_path = out_path(cfg, "roundtrip_outputs.jsonl")
    if not pred_path.exists() or not text_path.exists():
        return None, None, None, None, None

    pred = np.load(pred_path)
    pred_ids = [str(x) for x in pred["activation_ids"].tolist()]
    if pred_ids != ids:
        return None, None, None, None, None
    text_rows = read_jsonl(text_path)
    if [r.get("activation_id") for r in text_rows] != ids:
        return None, None, None, None, None

    sft_texts = [str(r["av_sft_explanation"]) for r in text_rows]
    rerank_texts = [str(r["av_rerank_explanation"]) for r in text_rows]
    pred_sft = pred["pred_sft"].astype(np.float32)
    pred_rerank = pred["pred_rerank"].astype(np.float32)
    return (
        sft_texts,
        rerank_texts,
        compute_metrics(h_test, pred_sft, h_train),
        compute_metrics(h_test, pred_rerank, h_train),
        pred_rerank,
    )


def main() -> None:
    cfg = parse_config()
    ensure_dirs(cfg)
    meta, h = read_vectors(cfg)
    train_idx = split_indices(meta, "train")
    test_idx = split_indices(meta, "test")
    h_train = subset(h, train_idx)
    h_test = subset(h, test_idx)
    rows_test = [meta[i] for i in test_idx]
    ids = [r["activation_id"] for r in rows_test]
    roles = [r["token_role"] for r in rows_test]
    ar_eval = load_ar(cfg, ar_artifact_path(cfg, "ar_eval"))
    av_rerank = None
    bootstrap = load_bootstrap_map(cfg)
    bootstrap_texts = [bootstrap[aid] for aid in ids]
    context_texts = [make_bootstrap_text(r, int(cfg["bootstrap_text"]["max_tokens"])) for r in rows_test]
    sft_texts, rerank_texts, sft_m, rerank_m, pred_rerank = _cached_roundtrip(cfg, ids, h_test, h_train)
    if sft_texts is None or rerank_texts is None or sft_m is None or rerank_m is None or pred_rerank is None:
        av_sft = load_av(cfg, av_artifact_path(cfg, "av_sft"))
        av_rerank = load_av(cfg, av_artifact_path(cfg, "av_rerank"))
        sft_texts = av_sft.generate(h_test, ids, mode="sft")
        rerank_texts = av_rerank.generate(h_test, ids, mode="rerank")
        sft_m = evaluate_text_method(sft_texts, h_test, h_train, ar_eval)
        pred_rerank = ar_eval.predict(rerank_texts)
        rerank_m = compute_metrics(h_test, pred_rerank, h_train)
    mean_pred, mean_m = mean_predictor(h_test, h_train)
    no_injection_m = _load_existing_method_metrics(cfg, "No-injection AV")
    if no_injection_m is None:
        if av_rerank is None:
            av_rerank = load_av(cfg, av_artifact_path(cfg, "av_rerank"))
        no_injection_m = no_injection_metrics(av_rerank, h_test, ids, h_train, ar_eval)
    methods = {
        "Mean predictor": mean_m,
        "Shuffled AV text": shuffled_text_metrics(rerank_texts, h_test, h_train, ar_eval, int(cfg.get("seed", 17))),
        "Role-preserving shuffled AV text": role_preserving_shuffled_text_metrics(
            rerank_texts, roles, h_test, h_train, ar_eval, int(cfg.get("seed", 17)) + 101
        ),
        "No-injection AV": no_injection_m,
        "AV-SFT -> AR_eval": sft_m,
        "AV-RerankSFT -> AR_eval": rerank_m,
    }
    if context_texts == bootstrap_texts:
        methods["Deterministic template/bootstrap text -> AR_eval"] = evaluate_text_method(bootstrap_texts, h_test, h_train, ar_eval)
        template_note = "context-only and bootstrap texts are identical deterministic templates"
    else:
        methods["Context-only pseudo-text -> AR_eval"] = evaluate_text_method(context_texts, h_test, h_train, ar_eval)
        methods["Bootstrap text -> AR_eval"] = evaluate_text_method(bootstrap_texts, h_test, h_train, ar_eval)
        template_note = "context-only and bootstrap texts differ"
    ci = bootstrap_ci_by_function(rows_test, h_test, pred_rerank, h_train, int(cfg["metrics"].get("bootstrap_samples", 200)), int(cfg.get("seed", 17)))
    payload = {
        "methods": methods,
        "ci_method": "AV-RerankSFT -> AR_eval",
        "ci_95": ci,
        "notes": {"mean_predictor_shape": list(mean_pred.shape), "template_baseline": template_note},
    }
    target = out_path(cfg, "metrics_main.json")
    write_json(target, payload)
    try:
        display = target.relative_to(Path.cwd())
    except ValueError:
        display = target
    print(f"wrote {display} with all requested baselines")


if __name__ == "__main__":
    main()
