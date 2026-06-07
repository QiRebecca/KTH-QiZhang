# Metric Definitions

Let $h_i$ be a held-out target activation and $\hat{h}_i$ be the AR
reconstruction from generated text. Let $\bar{h}_{\mathrm{train}}$ be the mean
raw activation over the train split.

Raw fraction of variance explained:

$$
\mathrm{FVE}_{\mathrm{raw}}
= 1 -
\frac{
  \sum_{i=1}^{n} \left\lVert h_i - \hat{h}_i \right\rVert_2^2
}{
  \sum_{i=1}^{n} \left\lVert h_i - \bar{h}_{\mathrm{train}} \right\rVert_2^2
}.
$$

This uses train-split statistics in the denominator and is evaluated on the
test split. It measures magnitude plus direction. It can be negative.

For directional metrics, define:

$$
u_i = \frac{h_i}{\left\lVert h_i \right\rVert_2},
\qquad
v_i = \frac{\hat{h}_i}{\left\lVert \hat{h}_i \right\rVert_2}.
$$

The directional train mean is:

$$
\bar{u}_{\mathrm{train}}
= \frac{1}{m}\sum_{j=1}^{m}
\frac{h^{\mathrm{train}}_j}{\left\lVert h^{\mathrm{train}}_j \right\rVert_2}.
$$

Cosine and normalized MSE:

$$
\mathrm{cosine}
= \frac{1}{n}\sum_{i=1}^{n}\left\langle u_i, v_i \right\rangle .
$$

$$
\mathrm{MSE}_{\mathrm{nrm}}
= \frac{1}{n}\sum_{i=1}^{n}\left\lVert u_i - v_i \right\rVert_2^2
= 2\left(1-\mathrm{cosine}\right).
$$

Directional FVE:

$$
\mathrm{FVE}_{\mathrm{dir}}
= 1 -
\frac{
  \sum_{i=1}^{n}\left\lVert u_i - v_i \right\rVert_2^2
}{
  \sum_{i=1}^{n}\left\lVert u_i - \bar{u}_{\mathrm{train}} \right\rVert_2^2
}.
$$

## Mean Predictor

The "Mean predictor" row repeats the raw train mean for every test sample. This
is exactly the denominator baseline for $\mathrm{FVE}_{\mathrm{raw}}$, so its
$\mathrm{FVE}_{\mathrm{raw}}$ is zero up to numerical precision.

It is not the denominator baseline for $\mathrm{FVE}_{\mathrm{dir}}$.
Directional scoring normalizes both target and prediction before measuring
error. Normalizing the raw train mean changes the geometry, so the
mean-predictor row can have negative $\mathrm{FVE}_{\mathrm{dir}}$.

## Token-role FVE

For token-role breakdowns, each role has its own numerator and denominator:

$$
\mathrm{FVE}_{\mathrm{raw}}(r)
= 1 -
\frac{\mathrm{SSE}_{\mathrm{raw}}(r)}
     {\mathrm{DEN}_{\mathrm{raw}}(r)} .
$$

The denominator still uses the train-split raw mean, but it is summed only over
examples in that role. This means role-level FVE values are diagnostic within a
role; they should not be sample-count averaged and compared directly with the
global FVE.

The correct decomposition is denominator-weighted:

$$
\mathrm{FVE}_{\mathrm{raw}}^{\mathrm{global}}
= 1 -
\frac{\sum_r \mathrm{SSE}_{\mathrm{raw}}(r)}
     {\sum_r \mathrm{DEN}_{\mathrm{raw}}(r)} .
$$

The audit writes `artifacts/role_fve_raw_denominator_breakdown.json` with:

- `raw_den_share`: the role's share of global raw denominator.
- `raw_sse_share`: the role's share of global raw SSE.
- `fve_raw_from_sse`: the role FVE recomputed from its SSE and denominator.
- directional equivalents for $\mathrm{FVE}_{\mathrm{dir}}$.
- denominator-weighted and sample-weighted role aggregations.

This is why the table can show positive raw-FVE values for several roles while
the global $\mathrm{FVE}_{\mathrm{raw}}$ is near zero. The global metric is
dominated by the roles and examples that carry most denominator mass and
reconstruction error, not by an unweighted average of role rows.

In the reported run, this caveat is not hypothetical. The `keyword` role has
only 52 test examples but accounts for about 98.86% of the raw-FVE denominator
and 99.18% of raw SSE. As a result, a sample-weighted average of role raw-FVEs
is positive while the true global raw FVE is near zero. The README therefore
uses role-level cosine, $\mathrm{MSE}_{\mathrm{nrm}}$, and
$\mathrm{FVE}_{\mathrm{dir}}$ for interpretation, and treats role-level raw FVE
as an audited diagnostic rather than the headline result.
