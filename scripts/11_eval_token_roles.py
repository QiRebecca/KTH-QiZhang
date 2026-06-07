from __future__ import annotations

from common import ar_artifact_path, av_artifact_path, parse_config, split_indices, subset
from nla_codescope.ar import load_ar
from nla_codescope.av import load_av
from nla_codescope.metrics import compute_metrics
from nla_codescope.utils import ensure_dirs, out_path, read_vectors, write_json
import numpy as np


def main() -> None:
    cfg = parse_config()
    ensure_dirs(cfg)
    meta, h = read_vectors(cfg)
    train_idx = split_indices(meta, "train")
    test_idx = split_indices(meta, "test")
    h_train = subset(h, train_idx)
    h_test = subset(h, test_idx)
    rows_test = [meta[i] for i in test_idx]
    ids = [r["activation_id"] for r in rows_test]
    pred_path = out_path(cfg, "roundtrip_predictions.npz")
    if pred_path.exists():
        pred_data = np.load(pred_path)
        saved_ids = [str(x) for x in pred_data["activation_ids"].tolist()]
        if saved_ids != ids:
            raise ValueError("roundtrip_predictions.npz activation_ids do not match current test split")
        pred = pred_data["pred_rerank"].astype(np.float32)
    else:
        ar_eval = load_ar(cfg, ar_artifact_path(cfg, "ar_eval"))
        av = load_av(cfg, av_artifact_path(cfg, "av_rerank"))
        texts = av.generate(h_test, ids, mode="rerank")
        pred = ar_eval.predict(texts)
    out = {}
    for role in sorted({r["token_role"] for r in rows_test}):
        idx = [i for i, r in enumerate(rows_test) if r["token_role"] == role]
        out[role] = compute_metrics(h_test[idx], pred[idx], h_train)
        out[role]["n"] = len(idx)
    write_json(out_path(cfg, "metrics_by_token_role.json"), out)
    print("wrote token-role metrics")


if __name__ == "__main__":
    main()
