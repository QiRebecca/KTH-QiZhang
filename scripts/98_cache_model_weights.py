from __future__ import annotations

from common import parse_config
from nla_codescope.runtime import configure_runtime_env


def main() -> None:
    cfg = parse_config()
    configure_runtime_env(cfg)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"].get("local_path") or cfg["model"]["name"]
    dtype = torch.bfloat16 if cfg["model"].get("dtype") == "bfloat16" else torch.float32
    local_only = bool(cfg["model"].get("local_files_only", False))
    print(f"Caching/full-loading model weights for {model_name}")
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_only)
    AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    print("Model weights available.")


if __name__ == "__main__":
    main()
