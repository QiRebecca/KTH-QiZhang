from __future__ import annotations

import yaml

from common import av_artifact_path, load_bootstrap_map, parse_config, split_indices, subset
from nla_codescope.av import train_lora_av, train_nearest_text_av
from nla_codescope.runtime import configure_runtime_env, configure_torch_runtime
from nla_codescope.utils import ensure_dirs, out_path, read_vectors


def main() -> None:
    cfg = parse_config()
    configure_runtime_env(cfg)
    if cfg.get("runtime", {}).get("require_cuda"):
        print(f"runtime: {configure_torch_runtime(cfg)}")
    ensure_dirs(cfg)
    if cfg["av"].get("backend") != "nearest_text_smoke":
        meta, h = read_vectors(cfg)
        texts = load_bootstrap_map(cfg)
        idx = split_indices(meta, "train")
        av = train_lora_av(subset(h, idx), [texts[meta[i]["activation_id"]] for i in idx], cfg, av_artifact_path(cfg, "av_sft"), int(cfg.get("seed", 17)))
        meta_path = out_path(cfg, "nla_meta_main.yaml")
        doc = yaml.safe_load(open(meta_path, "r", encoding="utf-8"))
        doc["injection_scale"] = av.injection_scale
        doc["injection_scale_selection"] = "median token embedding norm / median activation norm; A100 LoRA SFT path"
        yaml.safe_dump(doc, open(meta_path, "w", encoding="utf-8"), sort_keys=False)
        print(f"trained LoRA AV-SFT on {len(idx)} activations; injection_scale={av.injection_scale:.6f} -> {av.path}")
        return
    meta, h = read_vectors(cfg)
    texts = load_bootstrap_map(cfg)
    idx = split_indices(meta, "train")
    av = train_nearest_text_av(subset(h, idx), [texts[meta[i]["activation_id"]] for i in idx], cfg)
    av.save(av_artifact_path(cfg, "av_sft"))
    meta_path = out_path(cfg, "nla_meta_main.yaml")
    doc = yaml.safe_load(open(meta_path, "r", encoding="utf-8"))
    doc["injection_scale"] = av.injection_scale
    doc["injection_scale_selection"] = "smoke median token norm / median activation norm"
    yaml.safe_dump(doc, open(meta_path, "w", encoding="utf-8"), sort_keys=False)
    print(f"trained AV-SFT smoke model; injection_scale={av.injection_scale:.6f}")


if __name__ == "__main__":
    main()
