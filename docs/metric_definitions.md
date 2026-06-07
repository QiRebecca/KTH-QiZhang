# Metric Definitions

Let `h_i` be a held-out target activation and `hhat_i` be the AR reconstruction
from generated text. Let `mu_train` be the mean raw activation over the train
split.

Raw fraction of variance explained:

```text
FVE_raw = 1 - sum_i ||h_i - hhat_i||^2 / sum_i ||h_i - mu_train||^2
```

This uses train-split statistics in the denominator and is evaluated on the
test split. It measures magnitude plus direction. It can be negative.

For directional metrics, define:

```text
u_i = h_i / ||h_i||
v_i = hhat_i / ||hhat_i||
mu_dir_train = mean_j train_h_j / ||train_h_j||
```

Cosine and normalized MSE:

```text
cosine = mean_i <u_i, v_i>
MSE_nrm = mean_i ||u_i - v_i||^2
        = 2 * (1 - cosine)
```

Directional FVE:

```text
FVE_dir = 1 - sum_i ||u_i - v_i||^2 / sum_i ||u_i - mu_dir_train||^2
```

## Mean Predictor

The "Mean predictor" row repeats the raw train mean for every test sample. This
is exactly the denominator baseline for `FVE_raw`, so its `FVE_raw` is zero up
to numerical precision.

It is not the denominator baseline for `FVE_dir`. Directional scoring normalizes
both target and prediction before measuring error. Normalizing the raw train
mean changes the geometry, so the mean-predictor row can have negative
`FVE_dir`.

## Token-role FVE

For token-role breakdowns, each role has its own numerator and denominator:

```text
FVE_raw(role) = 1 - SSE_raw(role) / DEN_raw(role)
```

The denominator still uses the train-split raw mean, but it is summed only over
examples in that role. This means role-level FVE values are diagnostic within a
role; they should not be sample-count averaged and compared directly with the
global FVE.

The correct decomposition is denominator-weighted:

```text
global_FVE_raw = 1 - sum_role SSE_raw(role) / sum_role DEN_raw(role)
```

The audit writes `artifacts/role_fve_raw_denominator_breakdown.json` with:

- `raw_den_share`: the role's share of global raw denominator.
- `raw_sse_share`: the role's share of global raw SSE.
- `fve_raw_from_sse`: the role FVE recomputed from its SSE and denominator.
- directional equivalents for `FVE_dir`.
- denominator-weighted and sample-weighted role aggregations.

This is why the table can show positive raw-FVE values for several roles while
the global `FVE_raw` is near zero. The global metric is dominated by the roles
and examples that carry most denominator mass and reconstruction error, not by
an unweighted average of role rows.

In the reported run, this caveat is not hypothetical. The `keyword` role has
only 52 test examples but accounts for about 98.86% of the raw-FVE denominator
and 99.18% of raw SSE. As a result, a sample-weighted average of role raw-FVEs
is positive while the true global raw FVE is near zero. The README therefore
uses role-level cosine, `MSE_nrm`, and `FVE_dir` for interpretation, and treats
role-level raw FVE as an audited diagnostic rather than the headline result.
