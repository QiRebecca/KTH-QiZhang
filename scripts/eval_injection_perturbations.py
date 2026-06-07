from __future__ import annotations

"""GPU diagnostic: compare AV outputs under perturbed activation injections.

This intentionally runs only on a subset. It requires AV/AR checkpoints and is
not part of the CPU-only artifact audit.
"""

import argparse

import numpy as np

from common import ar_artifact_path, av_artifact_path, split_indices, subset
from nla_codescope.ar import load_ar
from nla_codescope.av import load_av
from nla_codescope.metrics import compute_metrics
from nla_codescope.utils import l2_normalize, load_config, out_path, read_vectors, write_json


def _components(h: np.ndarray, pred: np.ndarray, h_train: np.ndarray) -> dict[str, np.ndarray]:
    h64 = h.astype(np.float64)
    p64 = pred.astype(np.float64)
    train64 = h_train.astype(np.float64)
    h_norm = l2_normalize(h64)
    p_norm = l2_normalize(p64)
    mean_train_norm = l2_normalize(train64).mean(axis=0, keepdims=True)
    cosine = np.sum(h_norm * p_norm, axis=1)
    mse_nrm = np.sum((h_norm - p_norm) ** 2, axis=1)
    return {
        "cosine": cosine.astype(np.float64),
        "MSE_nrm": mse_nrm.astype(np.float64),
        "dir_num": mse_nrm.astype(np.float64),
        "dir_den": np.sum((h_norm - mean_train_norm) ** 2, axis=1).astype(np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 17))
    rng = np.random.default_rng(seed)
    meta, h = read_vectors(cfg)
    train_idx = split_indices(meta, "train")
    test_idx = split_indices(meta, "test")
    test_idx = test_idx[: min(args.max_samples, len(test_idx))]
    h_train = subset(h, train_idx)
    h_test = subset(h, test_idx)
    ids = [meta[i]["activation_id"] for i in test_idx]
    function_ids = [meta[i]["function_id"] for i in test_idx]
    av = load_av(cfg, av_artifact_path(cfg, "av_rerank"))
    ar_eval = load_ar(cfg, ar_artifact_path(cfg, "ar_eval"))

    mean_vec = h_train.mean(axis=0, keepdims=True).astype(np.float32)
    gaussian = rng.normal(size=h_test.shape).astype(np.float32)
    gaussian = gaussian / np.maximum(np.linalg.norm(gaussian, axis=1, keepdims=True), 1e-12)
    gaussian *= np.median(np.linalg.norm(h_train, axis=1))
    shuffled = h_test.copy()
    rng.shuffle(shuffled)
    fixed = np.repeat(h_test[:1], h_test.shape[0], axis=0)
    variants = {
        "correct_activation": h_test,
        "zero_vector": np.zeros_like(h_test),
        "train_mean_vector": np.repeat(mean_vec, h_test.shape[0], axis=0),
        "shuffled_activation": shuffled,
        "gaussian_norm_matched": gaussian,
        "fixed_first_activation": fixed,
    }
    out = {}
    components: dict[str, list[np.ndarray]] = {"cosine": [], "MSE_nrm": [], "dir_num": [], "dir_den": []}
    method_keys: list[str] = []
    for name, injected in variants.items():
        texts = av.generate(injected, ids, mode=f"perturb_{name}")
        pred = ar_eval.predict(texts)
        out[name] = compute_metrics(h_test, pred, h_train)
        c = _components(h_test, pred, h_train)
        method_keys.append(name)
        for k in components:
            components[k].append(c[k])
    write_json(out_path(cfg, "metrics_injection_perturbations.json"), out)
    np.savez_compressed(
        out_path(cfg, "per_sample_metrics_injection_perturbations.npz"),
        activation_ids=np.array(ids),
        function_ids=np.array(function_ids),
        method_keys=np.array(method_keys),
        cosine=np.stack(components["cosine"]),
        MSE_nrm=np.stack(components["MSE_nrm"]),
        dir_num=np.stack(components["dir_num"]),
        dir_den=np.stack(components["dir_den"]),
    )
    print(out)


if __name__ == "__main__":
    main()
