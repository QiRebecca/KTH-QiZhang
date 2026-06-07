from __future__ import annotations

"""Curate a small, reproducible qualitative-example set."""

import json

import numpy as np

from common import parse_config, split_indices
from nla_codescope.utils import out_path, read_jsonl, read_vectors, repo_root, write_jsonl


def _clip(text: str, n: int = 500) -> str:
    text = " ".join(str(text or "").split())
    return text[: n - 3] + "..." if len(text) > n else text


def _context(row: dict) -> str:
    for key in (
        "context_excerpt_for_human_display_only",
        "local_context_for_bootstrap_only",
        "local_context",
        "code",
    ):
        if row.get(key):
            return _clip(str(row[key]))
    return ""


def _pick(rows: list[dict], predicate, score_key: str, reverse: bool = True) -> dict:
    candidates = [r for r in rows if predicate(r)]
    if not candidates:
        candidates = rows
    return sorted(candidates, key=lambda r: float(r.get(score_key, 0.0)), reverse=reverse)[0]


def _pick_median(rows: list[dict], predicate, score_key: str) -> dict:
    candidates = [r for r in rows if predicate(r)]
    if not candidates:
        candidates = rows
    values = np.array([float(r.get(score_key, 0.0)) for r in candidates], dtype=np.float64)
    median = float(np.median(values))
    return sorted(candidates, key=lambda r: (abs(float(r.get(score_key, 0.0)) - median), r.get("activation_id", "")))[0]


def _pick_seeded_random(rows: list[dict], seed: int) -> dict:
    candidates = sorted(rows, key=lambda r: r.get("activation_id", ""))
    rng = np.random.default_rng(seed)
    return candidates[int(rng.integers(0, len(candidates)))]


def main() -> None:
    cfg = parse_config()
    meta, _ = read_vectors(cfg)
    test_rows = [meta[i] for i in split_indices(meta, "test")]
    meta_by_id = {r["activation_id"]: r for r in test_rows}
    output_rows = read_jsonl(out_path(cfg, "roundtrip_outputs.jsonl"))
    if not output_rows:
        raise FileNotFoundError("roundtrip_outputs.jsonl is required")
    enriched = []
    for r in output_rows:
        m = meta_by_id.get(r["activation_id"], {})
        row = {**m, **r}
        row["cosine"] = float(row.get("cosine_rerank", row.get("cosine", 0.0)))
        row["MSE_nrm"] = float(row.get("MSE_nrm_rerank", row.get("MSE_nrm", 0.0)))
        enriched.append(row)

    picks = [
        (
            "identifier_success",
            _pick(enriched, lambda r: r.get("token_role") == "function_name_or_identifier", "cosine", True),
            "Identifier success: highest-cosine identifier example under the deterministic selection policy.",
        ),
        (
            "operator_partial",
            _pick_median(enriched, lambda r: r.get("token_role") == "operator_or_punctuation", "cosine"),
            "Operator partial: operator/punctuation sample closest to the median cosine for that role.",
        ),
        (
            "literal_failure",
            _pick(enriched, lambda r: r.get("token_role") == "literal_string_or_number", "MSE_nrm", True),
            "Literal failure: highest-MSE literal example, illustrating loss of exact symbolic information.",
        ),
        (
            "generic_failure",
            _pick(enriched, lambda r: len(str(r.get("av_rerank_explanation", ""))) >= 120, "cosine", False),
            "Generic failure: low-cosine example with nontrivial generated text length.",
        ),
        (
            "seeded_random_check",
            _pick_seeded_random(enriched, int(cfg.get("seed", 17))),
            "Seeded random check: deterministic random test-set example for calibration.",
        ),
    ]

    curated = []
    for kind, row, interpretation in picks:
        curated.append(
            {
                "kind": kind,
                "activation_id": row.get("activation_id", ""),
                "function_id": row.get("function_id", ""),
                "token_role": row.get("token_role", ""),
                "target_token": row.get("token_text", ""),
                "code_context_for_human_inspection_only": _context(row),
                "av_generated_text": row.get("av_rerank_explanation", ""),
                "cosine": float(row.get("cosine", 0.0)),
                "MSE_nrm": float(row.get("MSE_nrm", 0.0)),
                "interpretation": interpretation,
            }
        )
    write_jsonl(out_path(cfg, "qualitative_examples_curated.jsonl"), curated)
    policy = {
        "seed": int(cfg.get("seed", 17)),
        "n_examples": len(curated),
        "policy": [
            "identifier_success: highest cosine among function_name_or_identifier samples",
            "operator_partial: operator_or_punctuation sample closest to median cosine",
            "literal_failure: highest MSE_nrm among literal_string_or_number samples",
            "generic_failure: lowest cosine with generated text length at least 120 characters",
            "seeded_random_check: deterministic random sample from sorted test activation ids",
        ],
        "not_random_performance_estimate": True,
    }
    out_path(cfg, "qualitative_selection_policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Qualitative Examples",
        "",
        "These examples are curated by deterministic rules to illustrate observed regimes; they are not a random estimate of overall performance.",
        "",
        "The code context is shown only for human inspection; it was not provided to the AV during final evaluation.",
        "",
    ]
    for row in curated:
        md.extend(
            [
                f"## {row['kind']}",
                "",
                f"- activation_id: `{row['activation_id']}`",
                f"- function_id: `{row['function_id']}`",
                f"- token_role: `{row['token_role']}`",
                f"- target_token: `{row['target_token']}`",
                f"- cosine: `{row['cosine']:.4f}`",
                f"- MSE_nrm: `{row['MSE_nrm']:.4f}`",
                f"- interpretation: {row['interpretation']}",
                "",
                "AV text:",
                "",
                "```text",
                _clip(row["av_generated_text"], 900),
                "```",
                "",
                "Human-only code context:",
                "",
                "```text",
                _clip(row["code_context_for_human_inspection_only"], 900),
                "```",
                "",
            ]
        )
    docs_path = repo_root() / "docs" / "qualitative_examples.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("\n".join(md), encoding="utf-8")
    print("wrote docs/qualitative_examples.md and artifacts/qualitative_examples_curated.jsonl")


if __name__ == "__main__":
    main()
