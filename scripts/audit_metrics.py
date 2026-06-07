from __future__ import annotations

"""Audit saved NLA metrics against saved targets/predictions when available.

This script is CPU-only if ``artifacts/roundtrip_predictions.npz`` exists.
For older runs that saved texts and aggregate JSON but not reconstructed
vectors, it emits a partial audit explaining which checks cannot be run.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from common import parse_config, split_indices, subset
from nla_codescope.metrics import compute_metrics, per_row_scores
from nla_codescope.utils import l2_normalize, out_path, read_vectors, write_json


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sse_raw(h: np.ndarray, pred: np.ndarray, train_h: np.ndarray) -> tuple[float, float, float]:
    mean_train = train_h.astype(np.float64).mean(axis=0, keepdims=True)
    h64 = h.astype(np.float64)
    p64 = pred.astype(np.float64)
    sse = float(np.sum((h64 - p64) ** 2))
    den = float(np.sum((h64 - mean_train) ** 2))
    return sse, den, float(1.0 - sse / max(den, 1e-12))


def _sse_dir(h: np.ndarray, pred: np.ndarray, train_h: np.ndarray) -> tuple[float, float, float]:
    train_norm = l2_normalize(train_h.astype(np.float64))
    h_norm = l2_normalize(h.astype(np.float64))
    p_norm = l2_normalize(pred.astype(np.float64))
    mean_train = train_norm.mean(axis=0, keepdims=True)
    sse = float(np.sum((h_norm - p_norm) ** 2))
    den = float(np.sum((h_norm - mean_train) ** 2))
    return sse, den, float(1.0 - sse / max(den, 1e-12))


def _check_or_raise(report: dict[str, Any]) -> None:
    failed: list[str] = []

    def visit(prefix: str, obj: Any) -> None:
        if isinstance(obj, bool):
            if not obj:
                failed.append(prefix)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                visit(f"{prefix}.{k}" if prefix else str(k), v)

    visit("", report.get("checks", {}))
    if failed:
        raise AssertionError("Metric audit failed checks: " + ", ".join(failed))


def main() -> None:
    cfg = parse_config()
    meta, h = read_vectors(cfg)
    train_idx = split_indices(meta, "train")
    test_idx = split_indices(meta, "test")
    h_train = subset(h, train_idx)
    h_test = subset(h, test_idx)
    rows_test = [meta[i] for i in test_idx]
    ids = [r["activation_id"] for r in rows_test]
    roles = [r["token_role"] for r in rows_test]

    report: dict[str, Any] = {
        "status": "partial",
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "d_model": int(h.shape[1]) if h.ndim == 2 else None,
        "role_counts_from_rows": {role: roles.count(role) for role in sorted(set(roles))},
        "checks": {},
        "limitations": [],
    }

    metrics_main = _load_json(out_path(cfg, "metrics_main.json"))
    role_metrics = _load_json(out_path(cfg, "metrics_by_token_role.json"))
    if role_metrics:
        role_n_sum = int(sum(int(v.get("n", 0)) for v in role_metrics.values()))
        report["checks"]["role_metric_n_sum_equals_test_n"] = role_n_sum == len(test_idx)
        report["role_counts_from_metrics"] = {k: int(v.get("n", 0)) for k, v in role_metrics.items()}
        report["checks"]["mse_nrm_equals_2_one_minus_cosine_by_role"] = {
            k: abs(float(v["MSE_nrm"]) - 2.0 * (1.0 - float(v["cosine"]))) < 1e-5
            for k, v in role_metrics.items()
            if "MSE_nrm" in v and "cosine" in v
        }
        weighted_by_n: dict[str, float] = {}
        for key in ("FVE_raw", "FVE_dir", "cosine", "MSE_nrm"):
            vals = [(float(v[key]), int(v.get("n", 0))) for v in role_metrics.values() if key in v]
            if vals and sum(n for _, n in vals):
                weighted_by_n[key] = float(sum(x * n for x, n in vals) / sum(n for _, n in vals))
        report["weighted_role_metrics_by_n"] = weighted_by_n

    method = "AV-RerankSFT -> AR_eval"
    if metrics_main.get("methods", {}).get(method):
        m = metrics_main["methods"][method]
        report["reported_global_method"] = method
        report["reported_global_metrics"] = m
        if "MSE_nrm" in m and "cosine" in m:
            report["checks"]["global_mse_nrm_equals_2_one_minus_cosine"] = (
                abs(float(m["MSE_nrm"]) - 2.0 * (1.0 - float(m["cosine"]))) < 1e-5
            )

    pred_path = out_path(cfg, "roundtrip_predictions.npz")
    if not pred_path.exists():
        report["limitations"].append(
            "roundtrip_predictions.npz is missing, so raw SSE, directional SSE, and exact FVE recomputation cannot be audited for this run. "
            "Regenerate from scripts/09_eval_roundtrip.py after the code update to enable full CPU-only auditing."
        )
        write_json(out_path(cfg, "metric_audit.json"), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if cfg.get("_require_complete"):
            raise SystemExit("metric audit is partial; roundtrip_predictions.npz is required")
        return

    pred_data = np.load(pred_path)
    saved_ids = [str(x) for x in pred_data["activation_ids"].tolist()]
    if saved_ids != ids:
        raise ValueError("roundtrip_predictions.npz activation_ids do not match current test split")
    pred = pred_data["pred_rerank"].astype(np.float32)
    report["status"] = "complete"
    raw_sse, raw_den, raw_fve = _sse_raw(h_test, pred, h_train)
    dir_sse, dir_den, dir_fve = _sse_dir(h_test, pred, h_train)
    recomputed = compute_metrics(h_test, pred, h_train)
    row_scores = per_row_scores(h_test, pred)
    report["global_recomputed"] = {
        **recomputed,
        "raw_sse": raw_sse,
        "raw_denominator": raw_den,
        "dir_sse": dir_sse,
        "dir_denominator": dir_den,
    }
    report["checks"]["raw_fve_matches_sse_formula"] = abs(recomputed["FVE_raw"] - raw_fve) < 1e-10
    report["checks"]["dir_fve_matches_sse_formula"] = abs(recomputed["FVE_dir"] - dir_fve) < 1e-10
    report["checks"]["per_row_mse_nrm_equals_2_one_minus_cosine"] = bool(
        np.allclose(row_scores["MSE_nrm"], 2.0 * (1.0 - row_scores["cosine"]), atol=1e-5)
    )

    role_details: dict[str, Any] = {}
    role_breakdown: dict[str, Any] = {}
    raw_sse_sum = raw_den_sum = dir_sse_sum = dir_den_sum = 0.0
    for role in sorted(set(roles)):
        idx = np.array([i for i, r in enumerate(roles) if r == role], dtype=np.int64)
        rsse, rden, rfve = _sse_raw(h_test[idx], pred[idx], h_train)
        dsse, dden, dfve = _sse_dir(h_test[idx], pred[idx], h_train)
        raw_sse_sum += rsse
        raw_den_sum += rden
        dir_sse_sum += dsse
        dir_den_sum += dden
        role_details[role] = {
            "n": int(len(idx)),
            "raw_sse": rsse,
            "raw_denominator_global_train_mean_group_sum": rden,
            "raw_den_share": float(rden / max(raw_den, 1e-12)),
            "raw_sse_share": float(rsse / max(raw_sse, 1e-12)),
            "FVE_raw": rfve,
            "fve_raw_from_sse": rfve,
            "dir_sse": dsse,
            "dir_denominator_global_train_mean_group_sum": dden,
            "dir_den_share": float(dden / max(dir_den, 1e-12)),
            "dir_sse_share": float(dsse / max(dir_sse, 1e-12)),
            "FVE_dir": dfve,
            "fve_dir_from_sse": dfve,
            "cosine": float(np.mean(row_scores["cosine"][idx])),
            "MSE_nrm": float(np.mean(row_scores["MSE_nrm"][idx])),
        }
        role_breakdown[role] = {
            "n": int(len(idx)),
            "raw_sse": rsse,
            "raw_den": rden,
            "raw_den_share": float(rden / max(raw_den, 1e-12)),
            "raw_sse_share": float(rsse / max(raw_sse, 1e-12)),
            "fve_raw_from_sse": rfve,
            "dir_sse": dsse,
            "dir_den": dden,
            "dir_den_share": float(dden / max(dir_den, 1e-12)),
            "fve_dir_from_sse": dfve,
            "cosine": float(np.mean(row_scores["cosine"][idx])),
            "MSE_nrm": float(np.mean(row_scores["MSE_nrm"][idx])),
        }
    report["role_recomputed"] = role_details
    denominator_weighted_raw = float(sum((v["raw_den_share"] * v["FVE_raw"]) for v in role_details.values()))
    denominator_weighted_dir = float(sum((v["dir_den_share"] * v["FVE_dir"]) for v in role_details.values()))
    sample_weighted_raw = float(sum((v["n"] * v["FVE_raw"]) for v in role_details.values()) / len(test_idx))
    sample_weighted_dir = float(sum((v["n"] * v["FVE_dir"]) for v in role_details.values()) / len(test_idx))
    report["role_fve_aggregation"] = {
        "denominator_weighted_role_fve_raw": denominator_weighted_raw,
        "global_minus_denominator_weighted_role_fve_raw": float(raw_fve - denominator_weighted_raw),
        "sample_weighted_role_fve_raw": sample_weighted_raw,
        "sample_weighted_minus_global_fve_raw": float(sample_weighted_raw - raw_fve),
        "denominator_weighted_role_fve_dir": denominator_weighted_dir,
        "global_minus_denominator_weighted_role_fve_dir": float(dir_fve - denominator_weighted_dir),
        "sample_weighted_role_fve_dir": sample_weighted_dir,
        "sample_weighted_minus_global_fve_dir": float(sample_weighted_dir - dir_fve),
        "warning": "Role FVE values must be denominator-weighted, not sample-weighted, to recover global FVE. Large denominator-share imbalance can make sample-weighted role raw FVE misleading.",
    }
    breakdown_payload = {
        "global": {
            "global_raw_sse": raw_sse,
            "global_raw_den": raw_den,
            "global_fve_raw": raw_fve,
            "global_dir_sse": dir_sse,
            "global_dir_den": dir_den,
            "global_fve_dir": dir_fve,
            **report["role_fve_aggregation"],
        },
        "roles": role_breakdown,
    }
    report["checks"]["sum_role_raw_sse_equals_global_raw_sse"] = abs(raw_sse_sum - raw_sse) / max(raw_sse, 1e-12) < 1e-8
    report["checks"]["sum_role_raw_den_equals_global_raw_den"] = abs(raw_den_sum - raw_den) / max(raw_den, 1e-12) < 1e-8
    report["checks"]["sum_role_dir_sse_equals_global_dir_sse"] = abs(dir_sse_sum - dir_sse) / max(dir_sse, 1e-12) < 1e-8
    report["checks"]["sum_role_dir_den_equals_global_dir_den"] = abs(dir_den_sum - dir_den) / max(dir_den, 1e-12) < 1e-8
    report["checks"]["denominator_weighted_role_fve_raw_equals_global_fve_raw"] = abs(denominator_weighted_raw - raw_fve) < 1e-10
    report["checks"]["denominator_weighted_role_fve_dir_equals_global_fve_dir"] = abs(denominator_weighted_dir - dir_fve) < 1e-10
    _check_or_raise(report)
    write_json(out_path(cfg, "role_fve_raw_denominator_breakdown.json"), breakdown_payload)
    write_json(out_path(cfg, "metric_audit.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
