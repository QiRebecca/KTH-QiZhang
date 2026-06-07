from __future__ import annotations

"""Audit saved NLA metrics against saved targets/predictions when available.

This script is CPU-only if ``artifacts/roundtrip_predictions.npz`` exists.
For older runs that saved texts and aggregate JSON but not reconstructed
vectors, it emits a partial audit explaining which checks cannot be run.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import parse_config, split_indices, subset
from nla_codescope.utils import out_path, read_vectors, write_json


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_close(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_json_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_close(x, y) for x, y in zip(a, b))
    return a == b


def _write_json_if_changed(path: Path, obj: Any) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if _json_close(existing, obj):
                return
        except Exception:
            pass
    write_json(path, obj)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _train_means(train_h: np.ndarray, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    raw_sum = np.zeros(train_h.shape[1], dtype=np.float64)
    norm_sum = np.zeros(train_h.shape[1], dtype=np.float64)
    n = 0
    for start in range(0, train_h.shape[0], chunk_size):
        x = train_h[start : start + chunk_size].astype(np.float64, copy=False)
        raw_sum += x.sum(axis=0)
        norm_sum += _normalize_rows(x).sum(axis=0)
        n += x.shape[0]
    return raw_sum / max(n, 1), norm_sum / max(n, 1)


def _per_row_components(
    h: np.ndarray,
    pred: np.ndarray,
    mean_train: np.ndarray,
    mean_train_norm: np.ndarray,
    chunk_size: int = 1024,
) -> dict[str, np.ndarray]:
    if h.shape != pred.shape:
        raise ValueError(f"h and prediction shape mismatch: {h.shape} vs {pred.shape}")
    n = h.shape[0]
    out = {
        "raw_num": np.empty(n, dtype=np.float64),
        "raw_den": np.empty(n, dtype=np.float64),
        "dir_num": np.empty(n, dtype=np.float64),
        "dir_den": np.empty(n, dtype=np.float64),
        "cosine": np.empty(n, dtype=np.float64),
        "MSE_nrm": np.empty(n, dtype=np.float64),
    }
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x = h[start:end].astype(np.float64, copy=False)
        y = pred[start:end].astype(np.float64, copy=False)
        x_norm = _normalize_rows(x)
        y_norm = _normalize_rows(y)
        raw_num = np.sum((x - y) ** 2, axis=1)
        raw_den = np.sum((x - mean_train) ** 2, axis=1)
        dir_num = np.sum((x_norm - y_norm) ** 2, axis=1)
        dir_den = np.sum((x_norm - mean_train_norm) ** 2, axis=1)
        cosine = np.sum(x_norm * y_norm, axis=1)
        out["raw_num"][start:end] = raw_num
        out["raw_den"][start:end] = raw_den
        out["dir_num"][start:end] = dir_num
        out["dir_den"][start:end] = dir_den
        out["cosine"][start:end] = cosine
        out["MSE_nrm"][start:end] = dir_num
    return out


def _aggregate_metrics(components: dict[str, np.ndarray]) -> dict[str, float]:
    raw_num = float(components["raw_num"].sum())
    raw_den = float(components["raw_den"].sum())
    dir_num = float(components["dir_num"].sum())
    dir_den = float(components["dir_den"].sum())
    return {
        "FVE_raw": float(1.0 - raw_num / max(raw_den, 1e-12)),
        "FVE_dir": float(1.0 - dir_num / max(dir_den, 1e-12)),
        "cosine": float(components["cosine"].mean()),
        "MSE_nrm": float(components["MSE_nrm"].mean()),
    }


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
        _write_json_if_changed(out_path(cfg, "metric_audit.json"), report)
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
    mean_train, mean_train_norm = _train_means(h_train)
    row_scores = _per_row_components(h_test, pred, mean_train, mean_train_norm)
    recomputed = _aggregate_metrics(row_scores)
    raw_sse = float(row_scores["raw_num"].sum())
    raw_den = float(row_scores["raw_den"].sum())
    raw_fve = float(1.0 - raw_sse / max(raw_den, 1e-12))
    dir_sse = float(row_scores["dir_num"].sum())
    dir_den = float(row_scores["dir_den"].sum())
    dir_fve = float(1.0 - dir_sse / max(dir_den, 1e-12))
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
        rsse = float(row_scores["raw_num"][idx].sum())
        rden = float(row_scores["raw_den"][idx].sum())
        rfve = float(1.0 - rsse / max(rden, 1e-12))
        dsse = float(row_scores["dir_num"][idx].sum())
        dden = float(row_scores["dir_den"][idx].sum())
        dfve = float(1.0 - dsse / max(dden, 1e-12))
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
    _write_json_if_changed(out_path(cfg, "role_fve_raw_denominator_breakdown.json"), breakdown_payload)
    _write_json_if_changed(out_path(cfg, "metric_audit.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
