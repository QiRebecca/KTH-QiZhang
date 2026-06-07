from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("Frozen code LM\nlayer-18 residual h", 0.08),
        ("AV\n<ACT> injection -> text", 0.34),
        ("AR\ntruncated LM + Linear", 0.60),
        ("h vs h_hat\nFVE / cosine", 0.84),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#333"))
    for x1, x2 in [(0.18, 0.27), (0.45, 0.53), (0.70, 0.78)]:
        ax.annotate("", xy=(x2, 0.55), xytext=(x1, 0.55), arrowprops=dict(arrowstyle="->", lw=1.8))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_main_bar(path: Path, metrics: dict[str, Any]) -> None:
    methods = list(metrics["methods"].keys())
    vals = [metrics["methods"][m]["FVE_dir"] for m in methods]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(methods)), vals, color="#4c78a8")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("FVE_dir")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
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
    roles = list(role_metrics.keys())
    vals = [role_metrics[r]["cosine"] for r in roles]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(roles)), vals, color="#59a14f")
    ax.set_ylabel("Cosine")
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
