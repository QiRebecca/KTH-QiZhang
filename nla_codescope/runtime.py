from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import Any


def nvidia_library_path() -> str:
    purelib = Path(sysconfig.get_paths()["purelib"])
    candidates = [
        purelib / "nvidia" / "cusparselt" / "lib",
        purelib / "nvidia" / "cusparse" / "lib",
        purelib / "nvidia" / "nccl" / "lib",
        purelib / "nvidia" / "cublas" / "lib",
        purelib / "nvidia" / "cudnn" / "lib",
        purelib / "nvidia" / "cuda_runtime" / "lib",
        purelib / "nvidia" / "cuda_nvrtc" / "lib",
        purelib / "nvidia" / "curand" / "lib",
        purelib / "nvidia" / "cusolver" / "lib",
    ]
    return ":".join(str(p) for p in candidates if p.exists())


def configure_runtime_env(cfg: dict[str, Any]) -> None:
    runtime = cfg.get("runtime", {})
    hf_home = runtime.get("hf_home")
    torch_home = runtime.get("torch_home")
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HF_DATASETS_CACHE", str(Path(hf_home) / "datasets"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))
    if torch_home:
        os.environ.setdefault("TORCH_HOME", torch_home)
    if runtime.get("cuda_alloc_conf"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", runtime["cuda_alloc_conf"])
    lib_path = nvidia_library_path()
    if lib_path:
        old = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = lib_path + (":" + old if old else "")


def configure_torch_runtime(cfg: dict[str, Any]) -> dict[str, Any]:
    import torch

    runtime = cfg.get("runtime", {})
    if runtime.get("allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    precision = runtime.get("matmul_precision")
    if precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)
    cuda_ok = torch.cuda.is_available()
    if runtime.get("require_cuda") and not cuda_ok:
        raise RuntimeError("Config requires CUDA, but torch.cuda.is_available() is false.")
    if cuda_ok and runtime.get("cuda_device") is not None:
        torch.cuda.set_device(int(runtime["cuda_device"]))
    return {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_ok,
        "device_name": torch.cuda.get_device_name() if cuda_ok else None,
        "device_capability": torch.cuda.get_device_capability() if cuda_ok else None,
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if cuda_ok else 0.0,
    }


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
