from __future__ import annotations

"""Write artifacts/manifest.json for reproducibility."""

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from common import parse_config, split_indices
from nla_codescope.utils import out_path, read_vectors, write_json


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.load(open(path, "r", encoding="utf-8"))


def main() -> None:
    cfg = parse_config()
    repo = Path(__file__).resolve().parents[1]
    config_path = Path(cfg.get("_config_path", "configs/main_qwen25coder_1p5b_L18.yaml"))
    try:
        config_rel = str(config_path.relative_to(repo))
    except ValueError:
        config_rel = "configs/main_qwen25coder_1p5b_L18.yaml"
    meta, h = read_vectors(cfg)
    train_idx = split_indices(meta, "train")
    val_idx = split_indices(meta, "val")
    test_idx = split_indices(meta, "test")
    function_counts = {}
    for split, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        function_counts[split] = len({meta[i]["function_id"] for i in idx})
    audit = maybe_json(out_path(cfg, "metric_audit.json"))
    artifact_names = [
        "metrics_main.json",
        "metrics_by_token_role.json",
        "metric_audit.json",
        "paired_delta_ci.json",
        "paired_delta_ci_all_baselines.json",
        "role_fve_raw_denominator_breakdown.json",
        "per_sample_metrics_main.npz",
        "per_sample_metrics_main_summary.json",
        "metrics_injection_perturbations.json",
        "per_sample_metrics_injection_perturbations.npz",
        "qualitative_examples.jsonl",
        "qualitative_examples_curated.jsonl",
        "qualitative_selection_policy.json",
        "roundtrip_outputs.jsonl",
        "roundtrip_predictions.npz",
        "activations.jsonl",
        "activations.npz",
        "bootstrap_texts.jsonl",
        "av_sft_outputs.jsonl",
        "av_rerank_best.jsonl",
        "nla_meta_main.yaml",
    ]
    artifacts = {}
    for name in artifact_names:
        p = out_path(cfg, name)
        if p.exists():
            artifacts[name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    manifest = {
        "project": "NLA-CodeScope",
        "run_type": "full_qwen_main",
        "model": cfg["model"]["name"],
        "d_model": int(h.shape[1]),
        "target_layer_label": cfg["model"]["target_layer_label"],
        "target_block_index_zero_based": cfg["model"]["target_block_index_zero_based"],
        "hidden_states_index": cfg["model"]["hidden_states_index"],
        "n_functions_or_activation_rows": len(meta),
        "dataset": cfg.get("data", {}).get("dataset_name"),
        "dataset_config": {
            "hf_dataset_name": cfg.get("data", {}).get("hf_dataset_name"),
            "hf_dataset_config": cfg.get("data", {}).get("hf_dataset_config"),
            "min_tokens": cfg.get("data", {}).get("min_tokens"),
            "max_tokens": cfg.get("data", {}).get("max_tokens"),
            "deduplicate_by_code_hash": cfg.get("data", {}).get("deduplicate_by_code_hash"),
            "activations_per_function": cfg.get("data", {}).get("activations_per_function"),
        },
        "train_functions": function_counts["train"],
        "val_functions": function_counts["val"],
        "test_functions": function_counts["test"],
        "split_sizes": {
            "train_functions": function_counts["train"],
            "val_functions": function_counts["val"],
            "test_functions": function_counts["test"],
        },
        "train_activations": len(train_idx),
        "val_activations": len(val_idx),
        "test_activations": len(test_idx),
        "activation_counts": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "n_train_activations": len(train_idx),
        "n_val_activations": len(val_idx),
        "n_test_activations": len(test_idx),
        "seed": cfg.get("seed", 17),
        "split_policy": cfg.get("data", {}).get("split_policy"),
        "final_av_eval_activation_only": True,
        "ar_train_used_for_rerank_only": True,
        "ar_eval_independent": True,
        "pending_gpu_tasks": [],
        "metric_audit_status": audit.get("status"),
        "av_lora": {k: cfg.get("av", {}).get(k) for k in ("lora_r", "lora_alpha", "lora_dropout")},
        "ar_lora": {k: cfg.get("ar", {}).get(k) for k in ("lora_r", "lora_alpha", "lora_dropout")},
        "runtime": {
            "gpu_used_for_main_run": "NVIDIA A100-PCIE-40GB",
            "main_run_log": "logs/a100_full_20260605_045246.log",
            "observed_runtime": "about 8h12m for the June 5 full run",
            "environment_spec": "environment.yml",
        },
        "manifest_generated_with": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "hardware": "NVIDIA A100-PCIE-40GB",
        "runtime_note": "Observed full run about 8h12m on one A100 40GB.",
        "gpu_runtime_summary": "One NVIDIA A100-PCIE-40GB; observed full run about 8h12m.",
        "commands_to_reproduce_metrics": [
            f"python scripts/audit_metrics.py --artifacts {cfg.get('output_dir', 'artifacts')} --require-complete",
            f"python scripts/reproduce_metrics_from_artifacts.py --artifacts {cfg.get('output_dir', 'artifacts')}",
            f"python scripts/bootstrap_delta_ci.py --config {config_rel}",
        ],
        "commands_to_reproduce_figures": [
            f"python scripts/reproduce_figures.py --artifacts {cfg.get('output_dir', 'artifacts')} --figures figures",
        ],
        "commands": [
            "bash scripts/run_main_pipeline.sh",
            f"python scripts/audit_metrics.py --config {config_rel}",
            f"python scripts/reproduce_figures.py --config {config_rel}",
        ],
        "artifacts": artifacts,
        "artifact_hashes": artifacts,
        "known_gpu_pending": maybe_json(out_path(cfg, "gpu_pending.json")),
        "notes": {
            "unknown_fields": [],
            "smoke_artifacts_location": "tests/fixtures/smoke/ and artifacts_smoke/ after running smoke tests",
        },
    }
    write_json(out_path(cfg, "manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
