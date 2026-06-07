from __future__ import annotations

import json
import os
import subprocess

from common import parse_config
from nla_codescope.runtime import configure_runtime_env, configure_torch_runtime


def main() -> None:
    cfg = parse_config()
    configure_runtime_env(cfg)
    probe_cfg = dict(cfg)
    probe_runtime = dict(cfg.get("runtime", {}))
    probe_runtime["require_cuda"] = False
    probe_cfg["runtime"] = probe_runtime
    info = configure_torch_runtime(probe_cfg)
    print(json.dumps(info, indent=2, sort_keys=True))
    if not info["cuda_available"]:
        raise SystemExit("CUDA is not available. Wait for the H800 allocation before running the main pipeline.")
    if "H800" not in (info["device_name"] or ""):
        print(f"Warning: expected an H800-class GPU, got {info['device_name']!r}.")
    if info["total_memory_gb"] < 70:
        raise SystemExit(f"GPU memory looks too small for the H800 profile: {info['total_memory_gb']} GB.")
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
        print(smi)
    except Exception as exc:
        print(f"nvidia-smi unavailable: {exc}")
    print("Environment paths:")
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "PYTORCH_CUDA_ALLOC_CONF"):
        print(f"{key}={os.environ.get(key, '')}")


if __name__ == "__main__":
    main()
