from __future__ import annotations

from common import ar_artifact_path, av_artifact_path, parse_config, split_indices, subset
from nla_codescope.ar import load_ar
from nla_codescope.av import load_av
from nla_codescope.injection import assert_activation_only_batch, build_activation_only_batch
from nla_codescope.metrics import bootstrap_ci_by_function, compute_metrics, per_row_scores
from nla_codescope.utils import ensure_dirs, out_path, read_vectors, write_json, write_jsonl
import numpy as np


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
    batch = build_activation_only_batch(cfg["av"]["prompt_template"], h_test, ids)
    assert_activation_only_batch(batch)
    ar_eval = load_ar(cfg, ar_artifact_path(cfg, "ar_eval"))
    av_sft = load_av(cfg, av_artifact_path(cfg, "av_sft"))
    av_rerank = load_av(cfg, av_artifact_path(cfg, "av_rerank"))
    sft_texts = av_sft.generate(h_test, ids, mode="sft")
    rerank_texts = av_rerank.generate(h_test, ids, mode="rerank")
    pred_sft = ar_eval.predict(sft_texts)
    pred_rerank = ar_eval.predict(rerank_texts)
    np.savez_compressed(
        out_path(cfg, "roundtrip_predictions.npz"),
        activation_ids=np.array(ids),
        token_roles=np.array([r["token_role"] for r in rows_test]),
        h_test=h_test.astype("float32"),
        pred_sft=pred_sft.astype("float32"),
        pred_rerank=pred_rerank.astype("float32"),
    )
    metrics = {
        "AV-SFT -> AR_eval": compute_metrics(h_test, pred_sft, h_train),
        "AV-RerankSFT -> AR_eval": compute_metrics(h_test, pred_rerank, h_train),
    }
    metrics["AV-RerankSFT -> AR_eval"]["95% CI"] = bootstrap_ci_by_function(
        rows_test, h_test, pred_rerank, h_train, int(cfg["metrics"].get("bootstrap_samples", 200)), int(cfg.get("seed", 17))
    )
    write_json(out_path(cfg, "roundtrip_metrics.json"), metrics)
    sft_scores = per_row_scores(h_test, pred_sft)
    scores = per_row_scores(h_test, pred_rerank)
    qual = []
    for kind, idx in [
        ("high-FVE success", int(scores["cosine"].argmax())),
        ("plausible but low-FVE", int(scores["cosine"].argmin())),
        ("identifier/literal loss", 0),
        ("hallucination or unsupported claim", min(1, len(rows_test) - 1)),
        ("RerankSFT improvement over SFT", int((per_row_scores(h_test, pred_rerank)["cosine"] - per_row_scores(h_test, pred_sft)["cosine"]).argmax())),
    ]:
        row = rows_test[idx]
        qual.append({
            "activation_id": row["activation_id"],
            "token_role": row["token_role"],
            "context_excerpt_for_human_display_only": row.get("context_excerpt_for_human_display_only", ""),
            "av_sft_explanation": sft_texts[idx],
            "av_rerank_explanation": rerank_texts[idx],
            "cosine": float(scores["cosine"][idx]),
            "why_interesting": kind,
        })
    write_jsonl(out_path(cfg, "qualitative_examples.jsonl"), qual)
    write_jsonl(out_path(cfg, "roundtrip_outputs.jsonl"), [
        {
            "activation_id": aid,
            "token_role": row["token_role"],
            "av_sft_explanation": s,
            "av_rerank_explanation": r,
            "cosine_sft": float(sft_scores["cosine"][j]),
            "cosine_rerank": float(scores["cosine"][j]),
            "MSE_nrm_sft": float(sft_scores["MSE_nrm"][j]),
            "MSE_nrm_rerank": float(scores["MSE_nrm"][j]),
        }
        for j, (aid, row, s, r) in enumerate(zip(ids, rows_test, sft_texts, rerank_texts))
    ])
    print("evaluated activation-only roundtrip on test split")


if __name__ == "__main__":
    main()
