from __future__ import annotations

"""Strict consistency checks for the final KTH submission artifacts."""

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
README = ROOT / "README.md"


EXPECTED_METRICS = {
    "Mean predictor": {"FVE_raw": 0.0000, "FVE_dir": -0.5102, "cosine": 0.5267, "MSE_nrm": 0.9465},
    "Shuffled AV text": {"FVE_raw": -0.0057, "FVE_dir": -0.3483, "cosine": 0.5775, "MSE_nrm": 0.8451},
    "Role-preserving shuffled AV text": {"FVE_raw": -0.0032, "FVE_dir": -0.2310, "cosine": 0.6142, "MSE_nrm": 0.7716},
    "No-injection AV": {"FVE_raw": -0.0061, "FVE_dir": -0.3872, "cosine": 0.5653, "MSE_nrm": 0.8695},
    "Deterministic template/bootstrap text -> AR_eval": {"FVE_raw": 0.0036, "FVE_dir": -0.1852, "cosine": 0.6286, "MSE_nrm": 0.7429},
    "AV-SFT -> AR_eval": {"FVE_raw": 0.0064, "FVE_dir": -0.1590, "cosine": 0.6368, "MSE_nrm": 0.7264},
    "AV-RerankSFT -> AR_eval": {"FVE_raw": -0.0004, "FVE_dir": -0.1243, "cosine": 0.6477, "MSE_nrm": 0.7046},
}

EXPECTED_COMPARISONS = [
    "AV-RerankSFT -> AR_eval minus AV-SFT -> AR_eval",
    "AV-RerankSFT -> AR_eval minus No-injection AV",
    "AV-RerankSFT -> AR_eval minus Shuffled AV text",
    "AV-RerankSFT -> AR_eval minus Role-preserving shuffled AV text",
    "AV-RerankSFT -> AR_eval minus Deterministic template/bootstrap text -> AR_eval",
]

EXPECTED_PERTURBATIONS = [
    "correct_activation",
    "zero_vector",
    "train_mean_vector",
    "shuffled_activation",
    "gaussian_norm_matched",
    "fixed_first_activation",
]


def load_json(name: str) -> dict[str, Any]:
    path = ART / name
    if not path.exists():
        raise AssertionError(f"missing required artifact: artifacts/{name}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_close(actual: float, expected: float, label: str, tol: float = 5e-4) -> None:
    if abs(float(actual) - expected) > tol:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def check_readme_metric_text(readme: str, expected: dict[str, dict[str, float]]) -> None:
    for method, vals in expected.items():
        if method not in readme:
            raise AssertionError(f"README missing metrics row/method text: {method}")
        for value in vals.values():
            formatted = f"{value:.4f}"
            if formatted not in readme and formatted.replace("-0.0004", "-0.0004") not in readme:
                raise AssertionError(f"README missing metric value {formatted} for {method}")


def main() -> None:
    errors: list[str] = []

    try:
        readme = README.read_text(encoding="utf-8")
        manifest = load_json("manifest.json")
        metrics = load_json("metrics_main.json")
        paired = load_json("paired_delta_ci_all_baselines.json")
        perturb = load_json("metrics_injection_perturbations.json")
        role = load_json("role_fve_raw_denominator_breakdown.json")
        audit = load_json("metric_audit.json")
        meta_path = ART / "nla_meta_main.yaml"
        if not meta_path.exists():
            raise AssertionError("missing required artifact: artifacts/nla_meta_main.yaml")
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

        if manifest.get("model") != "Qwen/Qwen2.5-Coder-1.5B-Instruct":
            raise AssertionError(f"manifest model is not Qwen: {manifest.get('model')}")
        if int(manifest.get("d_model", -1)) != 1536:
            raise AssertionError(f"manifest d_model is not 1536: {manifest.get('d_model')}")
        if manifest.get("run_type") not in {"full_qwen_main", "main_qwen_full"}:
            raise AssertionError(f"manifest run_type is not full_qwen_main: {manifest.get('run_type')}")
        pending = manifest.get("pending_gpu_tasks", manifest.get("known_gpu_pending", {}).get("pending_gpu_tasks"))
        if pending != []:
            raise AssertionError(f"manifest pending_gpu_tasks is not []: {pending}")
        if meta.get("target_model") != "Qwen/Qwen2.5-Coder-1.5B-Instruct":
            raise AssertionError(f"nla_meta_main target_model is not Qwen: {meta.get('target_model')}")
        if int(meta.get("d_model", -1)) != 1536:
            raise AssertionError(f"nla_meta_main d_model is not 1536: {meta.get('d_model')}")
        if audit.get("status") != "complete":
            raise AssertionError(f"metric_audit status is not complete: {audit.get('status')}")

        methods = metrics.get("methods", {})
        for method, vals in EXPECTED_METRICS.items():
            if method not in methods:
                raise AssertionError(f"metrics_main missing method: {method}")
            for metric, expected in vals.items():
                check_close(methods[method][metric], expected, f"{method} {metric}")
        check_readme_metric_text(readme, EXPECTED_METRICS)

        comparisons = paired.get("comparisons", {})
        for comparison in EXPECTED_COMPARISONS:
            if comparison not in comparisons:
                raise AssertionError(f"paired_delta_ci_all_baselines missing comparison: {comparison}")
        for condition in EXPECTED_PERTURBATIONS:
            if condition not in perturb:
                raise AssertionError(f"metrics_injection_perturbations missing condition: {condition}")

        keyword = role.get("roles", {}).get("keyword", {})
        check_close(keyword.get("raw_den_share"), 0.9886405865, "keyword raw_den_share", tol=1e-3)
        check_close(keyword.get("raw_sse_share"), 0.9918451051, "keyword raw_sse_share", tol=1e-3)

        server_path = "/" + "root" + "/" + "autodl-tmp"
        if server_path in readme:
            raise AssertionError("README contains server-local absolute path")
        lower = " ".join(readme.lower().split())
        forbidden = [
            "successfully " + "explains " + "activations",
            "upper bound for deterministic template",
            "promising " + "potential",
            "demonstrates " + "the power",
            "paves " + "the way",
            "significant " + "advancement",
            "causal " + "explanation",
        ]
        for phrase in forbidden:
            if phrase in lower:
                raise AssertionError(f"README contains overclaim phrase: {phrase}")
        for word in ("sota", "proves"):
            if re.search(rf"\b{re.escape(word)}\b", lower):
                raise AssertionError(f"README contains overclaim word: {word}")
        for phrase in ("faithful " + "explanation", "faithful " + "explanations", "model " + "thoughts"):
            if phrase in lower and f"not as {phrase}" not in lower and f"not as ground-truth {phrase}" not in lower:
                raise AssertionError(f"README contains non-negated overclaim phrase: {phrase}")

        for path in ART.iterdir():
            if path.is_file() and path.suffix in {".json", ".yaml", ".yml", ".jsonl"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if "smoke-tiny" in text:
                    raise AssertionError(f"main artifacts contain smoke metadata: {path}")
        smoke_fixture = ROOT / "tests" / "fixtures" / "smoke"
        if smoke_fixture.exists() and not (smoke_fixture / "artifacts").exists():
            raise AssertionError("tests/fixtures/smoke exists but does not clearly contain smoke artifacts")
    except AssertionError as exc:
        errors.append(str(exc))

    if errors:
        print("submission consistency check failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        raise SystemExit(1)
    print("submission consistency check passed")


if __name__ == "__main__":
    main()
