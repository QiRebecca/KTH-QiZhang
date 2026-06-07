from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    boxes = [
        ("Frozen code LM\nlayer-18 residual h", 0.10),
        ("AV\nactivation-only <ACT>\ninjection -> text", 0.36),
        ("AR\ntruncated LM + Linear\ntext -> h_hat", 0.62),
        ("Compare h vs h_hat\nFVE / cosine / MSE", 0.86),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#333"))
    for x1, x2 in [(0.20, 0.28), (0.46, 0.54), (0.72, 0.78)]:
        ax.annotate("", xy=(x2, 0.55), xytext=(x1, 0.55), arrowprops=dict(arrowstyle="->", lw=1.8))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_main_bar(path: Path, metrics: dict[str, Any]) -> None:
    label_map = {
        "AV-RerankSFT -> AR_eval": "AV-RerankSFT",
        "AV-SFT -> AR_eval": "AV-SFT",
        "Deterministic template/bootstrap text -> AR_eval": "Template/bootstrap",
        "Mean predictor": "Mean",
        "No-injection AV": "No injection",
        "Role-preserving shuffled AV text": "Role-shuffled",
        "Shuffled AV text": "Shuffled",
    }
    methods = sorted(metrics["methods"], key=lambda m: metrics["methods"][m]["FVE_dir"])
    vals = [metrics["methods"][m]["FVE_dir"] for m in methods]
    labels = [label_map.get(m, m) for m in methods]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(methods))
    ax.barh(y, vals, color="#4c78a8")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Directional FVE")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("All methods remain below zero; higher is better")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reconstruction_distribution(path: Path, scores: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, vals in scores.items():
        ax.hist(vals, bins=12, alpha=0.55, label=name)
    ax.set_xlabel("Per-example cosine")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_token_role_breakdown(path: Path, role_metrics: dict[str, Any]) -> None:
    label_map = {
        "function_name_or_identifier": "identifier",
        "operator_or_punctuation": "operator",
        "return_raise_yield_branch": "return/branch",
        "literal_string_or_number": "literal",
        "comment_or_docstring": "comment/docstring",
        "keyword": "keyword",
    }
    roles = sorted(role_metrics, key=lambda r: role_metrics[r]["cosine"])
    vals = [role_metrics[r]["cosine"] for r in roles]
    labels = [f"{label_map.get(r, r)}\n(n={int(role_metrics[r]['n'])})" for r in roles]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(range(len(roles)), vals, color="#59a14f")
    ax.set_ylabel("Cosine")
    ax.set_title("Reconstruction by token role")
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
