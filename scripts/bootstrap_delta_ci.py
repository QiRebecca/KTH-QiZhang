from __future__ import annotations

"""Paired bootstrap CIs for reconstruction metric improvements."""

import numpy as np

from common import parse_config, split_indices, subset
from nla_codescope.utils import l2_normalize, out_path, read_vectors, write_json


def paired_ci(rows: list[dict], h: np.ndarray, a: np.ndarray, b: np.ndarray, train_h: np.ndarray, samples: int, seed: int) -> dict:
    by_fn: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_fn.setdefault(row["function_id"], []).append(i)
    fns = sorted(by_fn)
    h_norm = l2_normalize(h.astype(np.float64))
    a_norm = l2_normalize(a.astype(np.float64))
    b_norm = l2_normalize(b.astype(np.float64))
    mean_train_norm = l2_normalize(train_h.astype(np.float64)).mean(axis=0, keepdims=True)
    dir_den_i = np.sum((h_norm - mean_train_norm) ** 2, axis=1)
    dir_num_a_i = np.sum((h_norm - a_norm) ** 2, axis=1)
    dir_num_b_i = np.sum((h_norm - b_norm) ** 2, axis=1)
    cosine_a_i = np.sum(h_norm * a_norm, axis=1)
    cosine_b_i = np.sum(h_norm * b_norm, axis=1)

    fn_dir_den = []
    fn_dir_num_a = []
    fn_dir_num_b = []
    fn_cosine_a = []
    fn_cosine_b = []
    fn_count = []
    for fn in fns:
        idx = np.array(by_fn[fn], dtype=np.int64)
        fn_dir_den.append(float(dir_den_i[idx].sum()))
        fn_dir_num_a.append(float(dir_num_a_i[idx].sum()))
        fn_dir_num_b.append(float(dir_num_b_i[idx].sum()))
        fn_cosine_a.append(float(cosine_a_i[idx].sum()))
        fn_cosine_b.append(float(cosine_b_i[idx].sum()))
        fn_count.append(int(len(idx)))
    dir_den = np.array(fn_dir_den, dtype=np.float64)
    dir_num_a = np.array(fn_dir_num_a, dtype=np.float64)
    dir_num_b = np.array(fn_dir_num_b, dtype=np.float64)
    cosine_a = np.array(fn_cosine_a, dtype=np.float64)
    cosine_b = np.array(fn_cosine_b, dtype=np.float64)
    counts = np.array(fn_count, dtype=np.float64)

    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {"delta_cosine": [], "delta_MSE_nrm": [], "delta_FVE_dir": []}
    n_fn = len(fns)
    for _ in range(samples):
        sampled = rng.integers(0, n_fn, size=n_fn)
        den = max(float(dir_den[sampled].sum()), 1e-12)
        count = max(float(counts[sampled].sum()), 1.0)
        a_num = float(dir_num_a[sampled].sum())
        b_num = float(dir_num_b[sampled].sum())
        vals["delta_cosine"].append(float((cosine_a[sampled].sum() - cosine_b[sampled].sum()) / count))
        vals["delta_MSE_nrm"].append(float((a_num - b_num) / count))
        vals["delta_FVE_dir"].append(float((b_num - a_num) / den))
    return {
        k: {
            "mean": float(np.mean(v)),
            "ci_95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
        }
        for k, v in vals.items()
    }


def paired_ci_from_components(
    function_ids: np.ndarray,
    a_dir_num: np.ndarray,
    b_dir_num: np.ndarray,
    dir_den: np.ndarray,
    a_cosine: np.ndarray,
    b_cosine: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    by_fn: dict[str, list[int]] = {}
    for i, fn in enumerate(function_ids.tolist()):
        by_fn.setdefault(str(fn), []).append(i)
    fns = sorted(by_fn)
    fn_a_num = []
    fn_b_num = []
    fn_den = []
    fn_a_cos = []
    fn_b_cos = []
    fn_count = []
    for fn in fns:
        idx = np.array(by_fn[fn], dtype=np.int64)
        fn_a_num.append(float(a_dir_num[idx].sum()))
        fn_b_num.append(float(b_dir_num[idx].sum()))
        fn_den.append(float(dir_den[idx].sum()))
        fn_a_cos.append(float(a_cosine[idx].sum()))
        fn_b_cos.append(float(b_cosine[idx].sum()))
        fn_count.append(int(len(idx)))
    a_num = np.array(fn_a_num, dtype=np.float64)
    b_num = np.array(fn_b_num, dtype=np.float64)
    den = np.array(fn_den, dtype=np.float64)
    a_cos = np.array(fn_a_cos, dtype=np.float64)
    b_cos = np.array(fn_b_cos, dtype=np.float64)
    counts = np.array(fn_count, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {"delta_cosine": [], "delta_MSE_nrm": [], "delta_FVE_dir": []}
    n_fn = len(fns)
    for _ in range(samples):
        sampled = rng.integers(0, n_fn, size=n_fn)
        total_den = max(float(den[sampled].sum()), 1e-12)
        total_count = max(float(counts[sampled].sum()), 1.0)
        a_total = float(a_num[sampled].sum())
        b_total = float(b_num[sampled].sum())
        vals["delta_cosine"].append(float((a_cos[sampled].sum() - b_cos[sampled].sum()) / total_count))
        vals["delta_MSE_nrm"].append(float((a_total - b_total) / total_count))
        vals["delta_FVE_dir"].append(float((b_total - a_total) / total_den))
    return {
        k: {
            "mean": float(np.mean(v)),
            "ci_95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
        }
        for k, v in vals.items()
    }


def _paired_from_npz(path, samples: int, seed: int) -> dict:
    data = np.load(path)
    methods = [str(x) for x in data["method_keys"].tolist()]
    method_idx = {m: i for i, m in enumerate(methods)}
    function_ids = data["function_ids"]
    target = "AV-RerankSFT -> AR_eval"
    controls = [
        "AV-SFT -> AR_eval",
        "No-injection AV",
        "Shuffled AV text",
        "Role-preserving shuffled AV text",
        "Deterministic template/bootstrap text -> AR_eval",
    ]
    out: dict = {
        "status": "complete",
        "bootstrap_unit": "function_id",
        "n_samples": int(len(function_ids)),
        "comparisons": {},
        "missing": [],
    }
    for control in controls:
        if target not in method_idx or control not in method_idx:
            out["missing"].append(f"{target} or {control} missing from per_sample_metrics_main.npz")
            continue
        a = method_idx[target]
        b = method_idx[control]
        out["comparisons"][f"{target} minus {control}"] = paired_ci_from_components(
            function_ids=function_ids,
            a_dir_num=data["dir_num"][a],
            b_dir_num=data["dir_num"][b],
            dir_den=data["dir_den"][a],
            a_cosine=data["cosine"][a],
            b_cosine=data["cosine"][b],
            samples=samples,
            seed=seed,
        )
    if out["missing"]:
        out["status"] = "partial"
    return out


def main() -> None:
    cfg = parse_config()
    meta, h = read_vectors(cfg)
    train_idx = split_indices(meta, "train")
    test_idx = split_indices(meta, "test")
    rows_test = [meta[i] for i in test_idx]
    h_train = subset(h, train_idx)
    h_test = subset(h, test_idx)
    pred_path = out_path(cfg, "roundtrip_predictions.npz")
    out: dict = {"status": "partial", "comparisons": {}, "missing": []}
    if not pred_path.exists():
        out["missing"].append("roundtrip_predictions.npz")
        write_json(out_path(cfg, "paired_delta_ci.json"), out)
        print(out)
        return
    pred = np.load(pred_path)
    samples = int(cfg.get("metrics", {}).get("bootstrap_samples", 1000))
    seed = int(cfg.get("seed", 17))
    per_sample_path = out_path(cfg, "per_sample_metrics_main.npz")
    if per_sample_path.exists():
        out = _paired_from_npz(per_sample_path, samples, seed)
        injection_path = out_path(cfg, "per_sample_metrics_injection_perturbations.npz")
        if injection_path.exists():
            # Injection perturbation uses the same component schema, with method
            # names specific to that diagnostic.
            inj = np.load(injection_path)
            methods = [str(x) for x in inj["method_keys"].tolist()]
            idx = {m: i for i, m in enumerate(methods)}
            for control in [
                "zero_vector",
                "train_mean_vector",
                "shuffled_activation",
                "gaussian_norm_matched",
                "fixed_first_activation",
            ]:
                if "correct_activation" in idx and control in idx:
                    out["comparisons"][f"correct_activation minus {control}"] = paired_ci_from_components(
                        function_ids=inj["function_ids"],
                        a_dir_num=inj["dir_num"][idx["correct_activation"]],
                        b_dir_num=inj["dir_num"][idx[control]],
                        dir_den=inj["dir_den"][idx["correct_activation"]],
                        a_cosine=inj["cosine"][idx["correct_activation"]],
                        b_cosine=inj["cosine"][idx[control]],
                        samples=samples,
                        seed=seed,
                    )
        else:
            out["missing"].append("per_sample_metrics_injection_perturbations.npz; cannot compute paired injection perturbation deltas")
            out["status"] = "partial"
        write_json(out_path(cfg, "paired_delta_ci_all_baselines.json"), out)
        write_json(out_path(cfg, "paired_delta_ci.json"), out)
        print(out)
        return

    out["comparisons"]["AV-RerankSFT minus AV-SFT"] = paired_ci(
        rows_test, h_test, pred["pred_rerank"].astype(np.float32), pred["pred_sft"].astype(np.float32), h_train, samples, seed
    )
    baseline_path = out_path(cfg, "baseline_predictions.npz")
    if not baseline_path.exists():
        out["missing"].append("baseline_predictions.npz; cannot compute paired deltas against shuffled/no-injection/template controls")
    out["status"] = "complete" if not out["missing"] else "partial"
    write_json(out_path(cfg, "paired_delta_ci.json"), out)
    print(out)


if __name__ == "__main__":
    main()
