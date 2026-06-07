from __future__ import annotations

"""Recompute or summarize metrics from saved artifacts without training.

Full recomputation is CPU-only when ``roundtrip_predictions.npz`` exists.
Older artifacts that only saved generated text are summarized and flagged.
"""

import json

from audit_metrics import main as audit_main
from common import parse_config
from nla_codescope.utils import out_path


def main() -> None:
    cfg = parse_config()
    audit_main()
    metrics_path = out_path(cfg, "metrics_main.json")
    if metrics_path.exists():
        payload = json.load(open(metrics_path, "r", encoding="utf-8"))
        print("\nMain metrics table:")
        print("method\tFVE_raw\tFVE_dir\tcosine\tMSE_nrm")
        for method, vals in payload.get("methods", {}).items():
            print(
                f"{method}\t{vals.get('FVE_raw', float('nan')):.6f}\t"
                f"{vals.get('FVE_dir', float('nan')):.6f}\t"
                f"{vals.get('cosine', float('nan')):.6f}\t"
                f"{vals.get('MSE_nrm', float('nan')):.6f}"
            )


if __name__ == "__main__":
    main()
