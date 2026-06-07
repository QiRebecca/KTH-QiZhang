from __future__ import annotations

import json
import os
import re
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
    runtime = cfg.get("runtime", {})
    profile = runtime.get("profile", "gpu")
    if not info["cuda_available"]:
        raise SystemExit(f"CUDA is not available. Wait for the {profile} allocation before running the main pipeline.")
    expected = runtime.get("expected_gpu_regex")
    if expected and not re.search(expected, info["device_name"] or "", re.IGNORECASE):
        print(f"Warning: expected GPU matching {expected!r}, got {info['device_name']!r}.")
    min_mem = float(runtime.get("min_cuda_memory_gb", 0))
    if info["total_memory_gb"] < min_mem:
        raise SystemExit(f"GPU memory is too small for {profile}: {info['total_memory_gb']} GB < {min_mem} GB.")
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
        print(smi)
    except Exception as exc:
        print(f"nvidia-smi unavailable: {exc}")
    print("Environment paths:")
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "PYTORCH_CUDA_ALLOC_CONF"):
        print(f"{key}={os.environ.get(key, '')}")
    if "A100" in (info["device_name"] or "") and info["total_memory_gb"] >= 75:
        print("A100 80GB detected: after the first successful run, consider extraction_batch_size=48-64 and generation_batch_size=24-32.")


if __name__ == "__main__":
    main()
