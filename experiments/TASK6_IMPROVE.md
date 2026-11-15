# Task 6: conditional book-date features and MAE ensembles

Accepted public reference: MAE 8,958,810.647357272 seconds. Retain all training events, including rating-zero events. Existing accepted notebook defaults remain unchanged pending public feedback.

## Features and models

Extend existing user-gap, rich history and book-date features with 29 conditional summaries of the supplied candidate book's known training interaction dates after the query user's latest date: future quantiles, counts within time windows, spread and user-pace-weighted date summaries. The supplied next-book ID is a legitimate test input. No test outcomes are used. Known training events of other users can occur after the query user's latest date in this offline setting.

For historical training examples, remove the target event itself from every book-date summary. For validation, globally exclude each user's final two events from the book-date corpus. Synthetic timestamp perturbations and direct sample recomputation check this exclusion.

Compare MAE LightGBM (31 leaves), recent-weighted LightGBM (63 leaves), user-balanced recent-weighted LightGBM (63 leaves), and MAE CatBoost (depth 6, user/book categories). The selected new ensemble uses balanced LightGBM and CatBoost. The primary adds the accepted forecast with nonnegative weights optimized on selection MAE only. Predict nonnegative gaps in seconds; training labels are internally in days.

## Chronological local results

192,034 earlier historical targets train models, 55,403 penultimate targets select settings/weights, and 55,403 final targets provide evaluation.

| Forecast | Evaluation MAE, seconds |
| --- | ---: |
| Accepted reference | 7,457,904.6219744105 |
| New-only ensemble | 7,422,607.322479249 |
| Primary blend | 7,419,901.322320926 |

Primary weights: accepted .1383320173, balanced LightGBM .4408620483, CatBoost .4208059344. New-only weights: balanced .4532685576, CatBoost .5467314424. Final models refit all 302,840 known targets. The local gain is modest and needs public confirmation.

## Reproduction

First `task6_improve.py --data-dir DATA --legacy CACHE --out OUT --full`, then `task6_improve_models.py` with the same arguments plus `--accepted ACCEPTED_CSV --predict` (omit `--full`). Requires NumPy, pandas, SciPy, scikit-learn, LightGBM and CatBoost. The legacy directory contains `task6_rich.npz` and `task6_final` features/models from the existing Task 6 pipeline; the accepted forecast is its cached output. This is an incremental experiment, not a standalone cache-free entrypoint.

Run `task1_6_checks.py --task 6` with matching paths to verify target exclusion, independently rebuilt features, model replay, nonnegative seconds and exact test IDs/order. See `task6_improvement_results.json` for hashes and audits. No datasets/model binaries are published. Only rejected Task 2 without-zero model caches were removed locally to free disk space; accepted models and datasets were retained.
