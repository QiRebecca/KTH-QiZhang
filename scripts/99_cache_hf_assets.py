from __future__ import annotations

from common import parse_config
from nla_codescope.runtime import configure_runtime_env


def main() -> None:
    cfg = parse_config()
    configure_runtime_env(cfg)
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    model_name = cfg["model"].get("local_path") or cfg["model"]["name"]
    local_only = bool(cfg["model"].get("local_files_only", False))
    print(f"Caching tokenizer/config for {model_name}")
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_only)
    AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_only)
    if cfg.get("data", {}).get("local_parquet_dir"):
        print("Skipping HuggingFace dataset cache because data.local_parquet_dir is configured.")
        return
    print("Caching CodeSearchNet metadata/data files")
    load_dataset(cfg["data"].get("hf_dataset_name", "code_search_net"), cfg["data"].get("hf_dataset_config", "python"))
    print("HF assets cached. Full model weights are intentionally left to the GPU stage unless you pass a stable HF mirror/token.")


if __name__ == "__main__":
    main()
