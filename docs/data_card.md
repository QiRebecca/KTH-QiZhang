# Data Card

## Source

The main experiment uses CodeSearchNet-style Python function-level data. The
configured preferred source is HuggingFace `code_search_net` with config
`python`; the pipeline also accepts a local parquet mirror with the same fields
when direct HuggingFace downloads are unavailable. The reported run was made
from a CodeSearchNet-compatible local parquet source, not from the synthetic
smoke dataset.

I did not independently re-audit the upstream repository licenses for every
function in this packaged artifact. The source should therefore be treated as a
CodeSearchNet-compatible research dataset rather than a newly licensed data
release. Repository/source identifiers are retained in `data/prepared_functions.jsonl`
when available.

The pipeline keeps these fields when available:

- `function_id`
- `repo` or `source_id`
- `code_hash`
- `code`
- `summary` or `docstring`
- `split`

## Filtering and Sampling

Functions are tokenized with the target `Qwen/Qwen2.5-Coder-1.5B-Instruct`
tokenizer and retained when the tokenized length is between 64 and 384 tokens.
Empty function bodies are removed. Code is deduplicated by `code_hash`.

Each retained function contributes four sampled non-special token positions:

- a deterministic random non-special token
- a function-name or identifier token if available
- a return/raise/yield/branch token if available
- an operator, literal, punctuation, indentation, or newline token if available

If a requested role is absent, the sampler falls back to a deterministic
non-special token. Token roles are heuristic and deterministic; see
[`nla_codescope/token_roles.py`](../nla_codescope/token_roles.py).

## Splits

The main run contains:

| split | functions | activations |
|---|---:|---:|
| train | 8,000 | 32,000 |
| validation | 1,000 | 4,000 |
| test | 1,000 | 4,000 |

Splits are by function id and code hash, not activation row. This prevents
activations from the same function from appearing in both train and test. The
leakage invariant is tested in
[`tests/test_split_no_leakage.py`](../tests/test_split_no_leakage.py).

## Bootstrap-only Context

The activation metadata includes `summary_text` and
`local_context_for_bootstrap_only` so deterministic SFT bootstrap texts can be
created. These fields are not passed to the AV during final roundtrip
evaluation. The final AV batch contains only `activation_id`, the fixed prompt,
and `activation_vector`; see
[`tests/test_av_no_context_eval.py`](../tests/test_av_no_context_eval.py).

## Smoke Dataset

[`configs/smoke.yaml`](../configs/smoke.yaml) uses a tiny synthetic Python
function dataset for CPU-friendly shape checks. It writes to `data_smoke/`,
`artifacts_smoke/`, and `figures_smoke/`. Smoke results are not used in the
README main table or claims.

## Known Limitations

The dataset is CodeSearchNet-style rather than a newly audited benchmark. Some
summaries/docstrings are noisy or mismatched, which weakens the deterministic
bootstrap targets. The token-role classifier is intentionally simple, so role
analysis is a diagnostic view rather than a ground-truth linguistic annotation.
The experiment evaluates reconstruction of activations, not code correctness or
causal behavior.
