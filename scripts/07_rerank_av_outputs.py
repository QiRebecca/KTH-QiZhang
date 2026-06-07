from __future__ import annotations

from common import ar_artifact_path, av_artifact_path, parse_config, split_indices, subset
from nla_codescope.ar import load_ar
from nla_codescope.av import load_av
from nla_codescope.rerank_sft import choose_best_candidates
from nla_codescope.utils import ensure_dirs, out_path, read_vectors, write_jsonl


def main() -> None:
    cfg = parse_config()
    ensure_dirs(cfg)
    meta, h = read_vectors(cfg)
    idx = split_indices(meta, "train")
    h_train = subset(h, idx)
    ids = [meta[i]["activation_id"] for i in idx]
    av = load_av(cfg, av_artifact_path(cfg, "av_sft"))
    ar_train = load_ar(cfg, ar_artifact_path(cfg, "ar_train"))
    candidates = av.sample_candidates(h_train, ids, int(cfg["av"]["rerank_samples_per_activation"]))
    best, scores = choose_best_candidates(ar_train, h_train, candidates)
    rows = [{"activation_id": aid, "best_text": text, "score": score} for aid, text, score in zip(ids, best, scores)]
    write_jsonl(out_path(cfg, "av_rerank_best.jsonl"), rows)
    print(f"selected AR_train best-of-N explanations for {len(rows)} train activations")


if __name__ == "__main__":
    main()
