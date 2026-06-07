# Reviewer Checklist

This is the submission self-check I use before packaging the repository.

| question | status | evidence |
|---|---|---|
| README is under 3000 words. | PASS | `wc -w README.md` reports about 1.5k words. |
| Final AV evaluation is activation-only. | PASS | README states the AV receives no source code, token text, function name, docstring, summary, local context, prefix, suffix, token role, or labels. `tests/test_av_no_context_eval.py` checks the final batch schema. |
| Layer convention is explicit. | PASS | `artifacts/nla_meta_main.yaml` records reported layer 18, zero-based block 17, and hidden-state index 18. |
| This is NLA rather than code summarization. | PASS | Final AV input is fixed prompt plus injected activation vector; context-derived text appears only in bootstrap/control rows. |
| `AR_train` and `AR_eval` are separated. | PASS | README and manifest state that `AR_train` is used only for reranking and independent `AR_eval` is used for final metrics. |
| Main baselines are present. | PASS | Main table includes mean, shuffled, role-preserving shuffled, no-injection, deterministic template/bootstrap, AV-SFT, and AV-RerankSFT. |
| Injection perturbation controls are present. | PASS | `artifacts/metrics_injection_perturbations.json` reports correct, zero, mean, shuffled, gaussian norm-matched, and fixed-first activation controls. |
| FVE formulas are clear. | PASS | `README.md` and `docs/metric_definitions.md` define `FVE_raw`, `FVE_dir`, cosine, and `MSE_nrm`. |
| Role/global raw-FVE caveat is handled. | PASS | README explains the denominator issue; `artifacts/role_fve_raw_denominator_breakdown.json` stores the audit values. |
| Negative results are interpreted honestly. | PASS | README says `FVE_raw` is near zero and `FVE_dir` remains negative; it describes the signal as weak reconstruction evidence, not explanation success. |
| Bootstrap texts are not called labels. | PASS | README calls them deterministic pseudo-texts used only for bootstrap training. |
| Qualitative examples include failures. | PASS | `docs/qualitative_examples.md` includes literal failure and generic failure cases selected by deterministic rules. |
| Smoke artifacts are isolated. | PASS | Smoke writes to `data_smoke/`, `artifacts_smoke/`, and `figures_smoke/`; full artifacts stay under `artifacts/`. |
| Artifact reproduction is CPU-friendly where possible. | PASS | `scripts/reproduce_metrics_from_artifacts.py` and `scripts/reproduce_figures.py` rebuild tables and figures from saved artifacts. |
| Large-file policy is checked. | PASS | No file currently exceeds 100 MB; `scripts/check_submission_ready.py` fails if a future file does. |
| Local file formats are parseable. | PASS | `scripts/validate_repository_format.py` compiles Python, checks shell syntax, parses YAML/TOML/JSON/JSONL, and detects line-collapsed Markdown/code. |
| Public raw GitHub view matches fresh clone. | PASS | `scripts/validate_public_repo.py` fresh-clones GitHub, downloads selected `raw.githubusercontent.com` files, compares SHA-256 hashes and line counts, and parses both views. |

## Remaining Acceptable Limitations

- `FVE_dir` remains negative and `FVE_raw` is near zero; this is reported as a weak signal rather than a success claim.
- RerankSFT is a best-of-4 approximation, not full reconstruction-driven RL.
- The deterministic bootstrap texts are noisy pseudo-texts, not ground-truth interpretations.
- The token-role classifier is deterministic but simple.
- This is reconstruction evidence only; it is not a causal intervention study.

## Commands

```bash
pytest -q
python3 scripts/validate_repository_format.py
python3 scripts/verify_submission_consistency.py
python3 scripts/audit_metrics.py --artifacts artifacts --require-complete
python3 scripts/reproduce_metrics_from_artifacts.py --artifacts artifacts
python3 scripts/reproduce_figures.py --artifacts artifacts --figures figures
python3 scripts/check_submission_ready.py
bash scripts/run_smoke_test.sh
python3 scripts/verify_submission_consistency.py
python3 scripts/check_submission_ready.py
python3 scripts/validate_public_repo.py --repo-url https://github.com/QiRebecca/KTH-QiZhang --branch main
```
