from __future__ import annotations

from common import av_artifact_path, parse_config
from nla_codescope.av import load_av
from nla_codescope.injection import assert_activation_only_batch, build_activation_only_batch
from nla_codescope.runtime import configure_runtime_env, configure_torch_runtime
from nla_codescope.utils import ensure_dirs, out_path, read_vectors, write_jsonl


def main() -> None:
    cfg = parse_config()
    configure_runtime_env(cfg)
    if cfg.get("runtime", {}).get("require_cuda"):
        print(f"runtime: {configure_torch_runtime(cfg)}")
    ensure_dirs(cfg)
    meta, h = read_vectors(cfg)
    av = load_av(cfg, av_artifact_path(cfg, "av_sft"))
    ids = [r["activation_id"] for r in meta]
    batch = build_activation_only_batch(cfg["av"]["prompt_template"], h, ids)
    assert_activation_only_batch(batch)
    texts = av.generate(h, ids, mode="sft")
    out = [{"activation_id": aid, "split": row["split"], "av_sft_explanation": text} for aid, row, text in zip(ids, meta, texts)]
    write_jsonl(out_path(cfg, "av_sft_outputs.jsonl"), out)
    print(f"generated {len(out)} activation-only AV-SFT outputs")


if __name__ == "__main__":
    main()
