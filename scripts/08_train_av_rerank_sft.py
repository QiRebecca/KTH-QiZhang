from __future__ import annotations

from common import av_artifact_path, parse_config, split_indices, subset
from nla_codescope.av import NearestTextAV, load_av, train_lora_av
from nla_codescope.runtime import configure_runtime_env, configure_torch_runtime
from nla_codescope.utils import ensure_dirs, out_path, read_jsonl, read_vectors


def main() -> None:
    cfg = parse_config()
    configure_runtime_env(cfg)
    if cfg.get("runtime", {}).get("require_cuda"):
        print(f"runtime: {configure_torch_runtime(cfg)}")
    ensure_dirs(cfg)
    if cfg["av"].get("backend") != "nearest_text_smoke":
        meta, h = read_vectors(cfg)
        idx = split_indices(meta, "train")
        best = {r["activation_id"]: r["best_text"] for r in read_jsonl(out_path(cfg, "av_rerank_best.jsonl"))}
        ids = [meta[i]["activation_id"] for i in idx]
        texts = [best[aid] for aid in ids]
        av = train_lora_av(
            subset(h, idx),
            texts,
            cfg,
            av_artifact_path(cfg, "av_rerank"),
            int(cfg.get("seed", 17)) + 2003,
            base_adapter_path=av_artifact_path(cfg, "av_sft"),
            rerank_texts=best,
        )
        print(f"trained LoRA AV-RerankSFT on {len(texts)} best-of-N texts -> {av.path}")
        return
    meta, h = read_vectors(cfg)
    idx = split_indices(meta, "train")
    best = {r["activation_id"]: r["best_text"] for r in read_jsonl(out_path(cfg, "av_rerank_best.jsonl"))}
    ids = [meta[i]["activation_id"] for i in idx]
    texts = [best[aid] for aid in ids]
    base = NearestTextAV.load(av_artifact_path(cfg, "av_sft"))
    av = NearestTextAV(train_h=subset(h, idx), train_texts=texts, injection_scale=base.injection_scale, rerank_texts=best)
    av.save(av_artifact_path(cfg, "av_rerank"))
    print(f"trained AV-RerankSFT smoke model on {len(texts)} best-of-N texts")


if __name__ == "__main__":
    main()
