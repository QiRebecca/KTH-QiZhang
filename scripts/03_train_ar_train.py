from __future__ import annotations

from common import ar_artifact_path, load_bootstrap_map, parse_config, split_indices, subset
from nla_codescope.ar import train_hashed_ridge, train_lora_ar
from nla_codescope.runtime import configure_runtime_env, configure_torch_runtime
from nla_codescope.utils import ensure_dirs, out_path, read_vectors


def main() -> None:
    cfg = parse_config()
    configure_runtime_env(cfg)
    if cfg.get("runtime", {}).get("require_cuda"):
        print(f"runtime: {configure_torch_runtime(cfg)}")
    ensure_dirs(cfg)
    if cfg["ar"].get("backend") != "hashed_ridge_smoke":
        meta, h = read_vectors(cfg)
        texts = load_bootstrap_map(cfg)
        idx = split_indices(meta, "train")
        model = train_lora_ar([texts[meta[i]["activation_id"]] for i in idx], subset(h, idx), cfg, ar_artifact_path(cfg, "ar_train"), int(cfg.get("seed", 17)))
        print(f"trained LoRA AR_train on {len(idx)} activations -> {model.path}")
        return
    meta, h = read_vectors(cfg)
    texts = load_bootstrap_map(cfg)
    idx = split_indices(meta, "train")
    model = train_hashed_ridge([texts[meta[i]["activation_id"]] for i in idx], subset(h, idx), cfg)
    model.save(ar_artifact_path(cfg, "ar_train"))
    print(f"trained AR_train on {len(idx)} activations")


if __name__ == "__main__":
    main()
