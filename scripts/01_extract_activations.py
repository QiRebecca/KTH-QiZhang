from __future__ import annotations

import yaml

from common import parse_config
from nla_codescope.extraction import extract_activations
from nla_codescope.runtime import configure_runtime_env, configure_torch_runtime
from nla_codescope.utils import data_path, ensure_dirs, out_path, read_jsonl, save_vectors


def main() -> None:
    cfg = parse_config()
    configure_runtime_env(cfg)
    if cfg.get("runtime", {}).get("require_cuda"):
        print(f"runtime: {configure_torch_runtime(cfg)}")
    ensure_dirs(cfg)
    rows = read_jsonl(data_path(cfg, "prepared_functions.jsonl"))
    meta, h = extract_activations(rows, cfg)
    save_vectors(cfg, meta, h)
    d_model = int(h.shape[1])
    meta_doc = {
        "target_model": cfg["model"]["name"],
        "target_layer_label": cfg["model"]["target_layer_label"],
        "target_block_index_zero_based": cfg["model"]["target_block_index_zero_based"],
        "hidden_states_index": cfg["model"]["hidden_states_index"],
        "d_model": d_model,
        "activation_type": "residual_stream_hidden_state",
        "injection_token": cfg["av"]["injection_token"],
        "injection_scale": None,
        "av_prompt_template": cfg["av"]["prompt_template"],
        "ar_architecture": f"truncate_to_layers={cfg['ar']['truncate_to_layers']}; value_head={cfg['ar']['value_head']}",
        "primary_metric": cfg["metrics"]["primary"],
        "secondary_metrics": cfg["metrics"]["secondary"],
        "split_policy": cfg["data"]["split_policy"],
        "seed": cfg.get("seed", 17),
    }
    with open(out_path(cfg, "nla_meta_main.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta_doc, f, sort_keys=False)
    print(f"saved {len(meta)} activations with d_model={d_model}")


if __name__ == "__main__":
    main()
