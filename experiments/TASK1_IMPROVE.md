# Task 1: SVD++, learned neighborhoods and nonlinear stacking

The accepted public reference has MSE 0.5538262939223092. This experiment excludes rating-zero rows. Accepted notebook defaults remain unchanged pending public feedback.

## Models

Retain the accepted SVD/cosine-neighborhood forecast. Add SVD++ at ranks 24 and 48, with regularization .1 and .15, learning rate .007 decayed by .985 per epoch, and checkpoints at 20/50/100 epochs. Its prediction is `mu + bu + bi + (Pu + sum(Yj)/sqrt(history_size)) dot Qi`. The C++ implementation performs exact cached implicit-vector updates for shuffled user blocks; it is independently checked against naive per-rating updates for that ordering. Cold factors are neutral.

Add six learned item-neighborhood regressions combining reading presence and rating residuals, using up to 64 neighbors and separate exposure/rating penalties. Self-book predictors are excluded. Two CatBoost stackers (depth 4/6) combine these signals with training-only user/item rating statistics. Their equal-weight ensemble is selected on an inner holdout.

## Validation

Positive observations: 1,834,017. Seed-42 split: 1,650,615 component-training observations, 91,701 selection observations and 91,701 untouched evaluation observations. Stacker depth/iterations/blending use an inner split within selection only; evaluation is reported afterward.

| Forecast | Local evaluation MSE |
| --- | ---: |
| Accepted reference | 0.5542632463064436 |
| New primary | 0.5384875114467605 |
| Half new / half accepted | 0.5422224312669193 |

Final components fit all positive observations; stackers refit on the combined held-out predictions/labels. Full predictions reuse the accepted CSV as a cached model forecast, never as a label. This avoids retraining the original three SVD components. Full-fit neighborhood signals reconstruct the SVD aggregate from that accepted forecast and its known linear weights.

## Reproduction

Run `task1_improve_stack.py --data-dir DATA --legacy CACHE --out OUT --accepted ACCEPTED_CSV --predict`. Requires NumPy, pandas, SciPy, CatBoost and g++. The legacy directory contains earlier validation prediction caches produced by the existing Task 1 experiment scripts; the accepted CSV can be regenerated with `task1_submission.py`. This is an incremental experiment, not a standalone cache-free entrypoint. Models and datasets remain local.

Run `task1_6_checks.py` with the same arguments plus `--task 1` to verify numerical updates, held-out graph exclusion, learned-parameter replay, cold factors, output IDs/order and finite rating ranges. See `task1_improvement_results.json` for hashes and audits. Local improvement does not establish a public or hidden-test gain.
