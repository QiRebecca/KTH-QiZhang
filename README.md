# Natural-language autoencoding of Qwen2.5-Coder activations

I reimplemented a small Natural Language Autoencoder (NLA) for residual-stream activations in `Qwen/Qwen2.5-Coder-1.5B-Instruct`. The target is layer 18 of 28 on Python function code. The final AV evaluation is activation-only: the verbalizer receives a fixed prompt plus an injected activation vector, with no source code, token text, function name, docstring, summary, local context, prefix, suffix, token role, or labels. The main result is mixed. AV-RerankSFT improves directional reconstruction over AV-SFT, no-injection, shuffled-text, role-preserving shuffled-text, and deterministic template controls, but `FVE_raw` remains near zero and `FVE_dir` remains negative. I interpret the generated text as weak reconstruction evidence: it carries measurable activation-specific directional information, but it is not a high-fidelity reconstruction channel. On code tokens, broad semantic/syntactic roles are more recoverable than exact literals, local control flow, tokenization detail, or activation magnitude.

## Research Question

Can natural language serve as a bottleneck for residual-stream activations of a code LLM? Where does that bottleneck preserve information about code tokens, and where does it fail?

This is a controlled small-model stress test of the NLA idea, not an Anthropic-scale RL reproduction and not an attempt to improve benchmark scores. The implementation has the paired NLA components: an activation verbalizer, AV, and an activation reconstructor, AR. The code path is in [`nla_codescope/`](nla_codescope/), with the final roundtrip evaluation in [`scripts/09_eval_roundtrip.py`](scripts/09_eval_roundtrip.py).

## Method

The target model is frozen. I extract a residual-stream hidden state `h`, inject it into the AV prompt at `<ACT>`, generate text, and reconstruct `h_hat` from that text with AR:

```text
Python function -> frozen Qwen2.5-Coder -> layer-18 residual h
h -> AV <ACT> embedding injection -> natural-language text
text -> AR truncated Qwen + Linear(d,d) -> h_hat
h vs h_hat -> FVE_raw / FVE_dir / cosine / MSE_nrm
```

Layer convention is fixed in [`artifacts/nla_meta_main.yaml`](artifacts/nla_meta_main.yaml): `target_layer_label = 18` for reporting, `target_block_index_zero_based = 17`, and HuggingFace extraction uses `hidden_states[18]` because `hidden_states[0]` is the embedding output. `d_model = 1536`.

The AV uses the same base model with a special `<ACT>` token whose embedding is replaced by `injection_scale * h`; see [`nla_codescope/injection.py`](nla_codescope/injection.py). The AR is a truncated 19-layer Qwen backbone with a final-token state and `Linear(d,d,bias=False)` value head. I train two ARs: `AR_train` is used only to score rerank candidates, while independent `AR_eval` is used for final metrics.

Bootstrap texts are deterministic pseudo-texts from summary/docstring, token role, and a short syntax hint. They bootstrap AV/AR training only; they are not ground-truth interpretations and are not provided to AV at final test time. RerankSFT is a bounded approximation to reconstruction-driven AV optimization: sample four AV explanations per train activation, score them with `AR_train`, keep the best, and train one more AV epoch. This is not full RL.

The activation-only invariant is tested in [`tests/test_av_no_context_eval.py`](tests/test_av_no_context_eval.py). The final evaluation batch contains only `activation_id`, fixed `prompt`, and `activation_vector`.

## Experimental Setup

The main run uses CodeSearchNet-style Python function data. See [`docs/data_card.md`](docs/data_card.md) for source and filtering details.

| split | functions | activations |
|---|---:|---:|
| train | 8,000 | 32,000 |
| validation | 1,000 | 4,000 |
| test | 1,000 | 4,000 |

Each function contributes four sampled non-special token positions. Splits are by function id and code hash, not activation row, so activations from the same function do not cross train/validation/test. The full run used LoRA rather than full fine-tuning and ran on one A100 40GB GPU for about 8.2 hours. The artifact manifest is [`artifacts/manifest.json`](artifacts/manifest.json).

## Metrics and Audit

Raw FVE uses the train-split raw mean:

```text
FVE_raw = 1 - sum_i ||h_i - hhat_i||^2 / sum_i ||h_i - mu_train||^2
```

For directional metrics:

```text
u_i = h_i / ||h_i||
v_i = hhat_i / ||hhat_i||
cosine = mean_i <u_i, v_i>
MSE_nrm = mean_i ||u_i - v_i||^2 = 2 * (1 - cosine)
FVE_dir = 1 - sum_i ||u_i - v_i||^2 / sum_i ||u_i - mu_dir_train||^2
```

FVE can be negative. `FVE_raw` scores magnitude and direction; `FVE_dir` scores normalized direction. The mean predictor is exactly the `FVE_raw` denominator baseline, but not the `FVE_dir` baseline because directional scoring normalizes the prediction before measuring error.

The metric audit in [`artifacts/metric_audit.json`](artifacts/metric_audit.json) is complete. It recomputes raw/directional SSE from [`artifacts/roundtrip_predictions.npz`](artifacts/roundtrip_predictions.npz), checks that role SSEs and denominators sum to global values, and verifies `MSE_nrm = 2 * (1 - cosine)`. Metric definitions are in [`docs/metric_definitions.md`](docs/metric_definitions.md).

One audit issue matters for interpretation. Role-level raw FVE can be misleading if averaged by sample count: `sample_weighted_role_fve_raw = +0.2793`, while global `FVE_raw = -0.00036`. The reason is denominator mass: the `keyword` group has only 52 examples but contributes 98.86% of the raw-FVE denominator and 99.18% of raw SSE. Denominator weighting recovers the global raw FVE. For token-role discussion I therefore emphasize cosine, `MSE_nrm`, and `FVE_dir`; the denominator breakdown is in [`artifacts/role_fve_raw_denominator_breakdown.json`](artifacts/role_fve_raw_denominator_breakdown.json).

## Main Results

| Method | FVE_raw | FVE_dir | Cosine | MSE_nrm |
|---|---:|---:|---:|---:|
| Mean predictor | 0.0000 | -0.5102 | 0.5267 | 0.9465 |
| Shuffled AV text | -0.0057 | -0.3483 | 0.5775 | 0.8451 |
| Role-preserving shuffled AV text | -0.0032 | -0.2310 | 0.6142 | 0.7716 |
| No-injection AV | -0.0061 | -0.3872 | 0.5653 | 0.8695 |
| Deterministic template/bootstrap text -> AR_eval | 0.0036 | -0.1852 | 0.6286 | 0.7429 |
| AV-SFT -> AR_eval | 0.0064 | -0.1590 | 0.6368 | 0.7264 |
| AV-RerankSFT -> AR_eval | -0.0004 | -0.1243 | 0.6477 | 0.7046 |

The deterministic template/bootstrap row is a context-derived pseudo-text control implemented by the same deterministic template used for bootstrap training. It is a prior/control, not an upper bound.

AV-RerankSFT has the best directional reconstruction among tested AV variants. Its 95% bootstrap CIs over function id are: `FVE_raw [-0.00135, 0.00169]`, `FVE_dir [-0.12785, -0.12065]`, cosine `[0.64523, 0.65015]`, and `MSE_nrm [0.69971, 0.70955]`. The main bar chart is [`figures/main_fve_bar.png`](figures/main_fve_bar.png): it shows the directional improvement over controls while all FVE_dir values remain negative.

Paired bootstrap deltas in [`artifacts/paired_delta_ci_all_baselines.json`](artifacts/paired_delta_ci_all_baselines.json) use function id as the resampling unit:

| paired comparison | delta FVE_dir | 95% CI |
|---|---:|---:|
| AV-RerankSFT vs AV-SFT | +0.0348 | [0.0311, 0.0387] |
| AV-RerankSFT vs no-injection AV | +0.2630 | [0.2591, 0.2669] |
| AV-RerankSFT vs shuffled AV text | +0.2240 | [0.2174, 0.2306] |
| AV-RerankSFT vs role-preserving shuffled text | +0.1067 | [0.1011, 0.1125] |
| AV-RerankSFT vs deterministic template/bootstrap | +0.0609 | [0.0567, 0.0650] |

An injection perturbation diagnostic on 1,000 held-out activations checks whether AV uses the vector rather than only a generic language prior:

| injection condition | FVE_dir | cosine |
|---|---:|---:|
| correct activation | -0.1239 | 0.6470 |
| zero vector | -0.3912 | 0.5630 |
| train mean vector | -0.2983 | 0.5922 |
| shuffled activation | -0.3556 | 0.5742 |
| gaussian norm-matched | -0.3793 | 0.5667 |
| fixed first activation | -0.4501 | 0.5445 |

Correct activation is clearly better than the perturbations, so the AV is using activation-specific information. The reconstruction distribution in [`figures/reconstruction_distribution.png`](figures/reconstruction_distribution.png) shows that this is not uniform across examples; many reconstructions remain weak.

## Token-role Analysis

| role | n | FVE_raw | FVE_dir | cosine |
|---|---:|---:|---:|---:|
| identifier | 1,337 | 0.3163 | -0.0476 | 0.6951 |
| operator/punctuation | 1,485 | 0.2658 | -0.1479 | 0.6567 |
| return/branch | 986 | 0.2676 | -0.1749 | 0.5858 |
| literal | 131 | 0.2560 | -0.1980 | 0.5826 |
| keyword | 52 | -0.0036 | -0.0460 | 0.5191 |
| comment/docstring | 9 | 0.2706 | -0.2000 | 0.5792 |

The pattern in [`figures/token_role_breakdown.png`](figures/token_role_breakdown.png) suggests that identifiers and operators are directionally reconstructed better than literals and return/branch/control-flow tokens. The comment/docstring group is too small to interpret. The code-domain lesson is narrow: natural language preserves broad semantic or syntactic role better than exact symbolic detail.

## Qualitative Examples

[`docs/qualitative_examples.md`](docs/qualitative_examples.md) contains five deterministic-rule examples: identifier high-cosine, median operator/punctuation, literal failure, generic failure, and a seeded random check. The selection policy is saved in [`artifacts/qualitative_selection_policy.json`](artifacts/qualitative_selection_policy.json). These examples illustrate regimes; they are not a random estimate of performance. The code context is shown only for human inspection and was not provided to the AV during final evaluation. Plausible text is not faithfulness.

## Limitations

`FVE_dir` remains negative and `FVE_raw` is near zero. The texts are reconstruction-constrained hypotheses, not direct interpretations. Reconstruction evidence is not an intervention; I do not run activation patching, steering, or code-correctness causal tests. RerankSFT is a best-of-4 SFT approximation, not Anthropic-scale RL. Deterministic bootstrap pseudo-texts are weak training targets, not labels. The experiment uses one small model and one target layer. Code correctness is not directly evaluated, and exact literals, local control flow, tokenization details, and magnitude remain weakly represented by the natural-language bottleneck.

## Reproducibility

Artifact-level metric and figure reproduction does not require GPU. Full extraction/training requires GPU and is controlled by the configs and pipeline scripts. Smoke tests use isolated `artifacts_smoke/`, `data_smoke/`, and `figures_smoke/` directories and do not overwrite the full-run artifacts.

```bash
pytest -q
python3 scripts/validate_repository_format.py
python3 scripts/verify_submission_consistency.py
python3 scripts/audit_metrics.py --artifacts artifacts --require-complete
python3 scripts/reproduce_metrics_from_artifacts.py --artifacts artifacts
python3 scripts/reproduce_figures.py --artifacts artifacts --figures figures
python3 scripts/check_submission_ready.py
bash scripts/run_smoke_test.sh
python3 scripts/validate_public_repo.py --repo-url https://github.com/QiRebecca/KTH-QiZhang --branch main
```

The largest included file is [`artifacts/activations.npz`](artifacts/activations.npz), about 90 MB. No artifact exceeds 100 MB in this repository state, so Git LFS is not required for the current files. If a hosting service rejects large regular files, track `artifacts/*.npz` with Git LFS before pushing.
