# Task 6: date-aware time-gap baseline

The user reports HR86 scores before this update: Task1 MSE 0.5538262939223092; Task2 accuracy 0.9461; Task3 MSE 0.704589533460822; Task4 accuracy 0.9403; Task5 accuracy 0.9093402161638974; Task6 MAE 13,130,029.77082897 seconds. Tasks 1-5 are retained.

## Model

Predict the gap between the user's last observed interaction and the next interaction with the book ID supplied in the test row. Features use only provided training observations and the test user/book IDs; the future test timestamp/delta is unavailable and never used.

The selected nonnegative MAE-optimized blend is:
- 88.5499% CatBoost trained on gap days with MAE loss.
- 9.5863% CatBoost trained on log1p(gap days), inverted with expm1.
- 1.8638% Ridge using last, median and mean historical gaps.

The two tree models have depth6, learning_rate .05, l2_leaf_reg3, seed42, and max_ctr_complexity=1. Their selected iteration counts are 1196 and 1171. The categorical-complexity cap prevents large, mostly unique user/book interaction encodings from producing impractical model files. Numeric input comprises 64 history features and 16 book-date features, plus categorical user and next-book IDs.

History features include recent log gaps, overall/recent quantiles, means and variability, log-gap EMA at several alphas, interaction count, observed-history length, last known date, seasonal signals and known rating-history summaries. These extend the slides' autoregression, sliding-window and recency-weighting ideas with nonlinear regression. CatBoost is an additional method, not claimed to be a specific lecture implementation.

Book-date features describe other available training observations of the next book: count, first/last/mean/median/quantile dates relative to the user's last date, variability, nearest past/future observed book event and activity after the user's last date. First observed dates are not asserted to be publication dates. Books absent from the metadata receive missing numerical summaries and an unknown-book indicator; CatBoost handles unseen categorical IDs.

## Leakage controls and chronology

All 55,403 users have at least four observed interactions. Per-user ordering is verified, not reconstructed from hidden targets. Hold out the final two observed events per user: penultimate gap for selection, final gap for evaluation. The original earlier-target sample has 192,034 examples, with up to five evenly spaced earlier targets per user. Each held-out partition has 55,403 examples. History features stop immediately before each target event.

Book-date summaries for these experiments are built from the earlier-prefix events only, excluding both held-out events of every user globally. For each earlier training example, its own target event is additionally removed from its book's statistics. This leave-one-event-out exclusion prevents the next timestamp from leaking through a book mean, median or nearest-future-date feature.

Other users' provided training dates can be later than a particular query user's last date; these are known observations in this offline assignment. This is not a strict live-deployment backtest with one global calendar cutoff. No selection/evaluation target dates enter the book-summary corpus. The future test event itself is never available.

The direct and log models' stopping iterations and blend weights use selection data only. Evaluation gaps are not fitted until the final production refit. The benchmark has been inspected in earlier exploratory rounds; it is not a pristine final competition test.

## Local evaluation (MAE in seconds)

| Predictor | MAE |
| --- | ---: |
| Median historical gap | 12,052,136 |
| Three-feature Ridge on the same sampled earlier targets | 11,436,786 |
| History + user/book IDs, controlled complexity | 9,538,405 |
| Earlier experimental ensemble | 9,433,913 |
| Date-aware direct MAE model | 7,468,395 |
| Date-aware log-gap model | 7,566,359 |
| Selected date-aware ensemble | **7,457,905** |

Selection MAE for the final blend is 7,231,698 seconds. The major gain is the book-date information, not simply averaging more models. Full comparisons and weights are in task6_results.json. The old GARCH experiment did not improve on the median and is not included.

These are local metrics. A new Gradescope submission is required to compare with HR86's prior 13.13 million MAE or a leaderboard target/rank.

## Full-history refit and prediction

For the final model, feature histories and book-date summaries use all provided training observations. Each historical training target still excludes its own book event. The training sample retains the five earlier targets plus each user's two final observed targets: 302,840 total examples. All history observations can contribute features/statistics, but not every possible historical gap is used as a separate target. This balances very active users and controls fit cost.

Output is 55,403 nonnegative finite deltas in **seconds**, preserving test user IDs, book IDs, row order and header. The reusable pipeline exactly reproduced the generated CSV. Target-row ordering and sampled regenerated history features matched the experimental caches exactly.

## Run

From repository root:

```bash
pip install numpy pandas scikit-learn catboost
python experiments/task6_checks.py
python experiments/task6_pipeline.py \
  --data-dir 'Assignment 1/assignment1_2026' \
  --output-dir experiments/artifacts/task6 \
  --predict
```

Use --evaluate to replay the frozen selected-model backtest, or both flags to evaluate and predict. The configuration fixes the selected iterations and weights; this command does not rerun the entire earlier exploratory search. Data/source/package fingerprints isolate caches. Fresh feature construction and fitting can take several minutes. The current Task 6 notebook cell invokes this pipeline and copies task_6_predictions.csv beside the inputs.

Synthetic leakage tests check that removing a target date changes the nearest-future summary correctly, a globally held-out event remains excluded, and a singleton book becomes unknown after self-removal. CSV checks verify all 55,403 IDs, rows, header, finite values and nonnegative seconds.

Source data and earlier delivered CSVs are unchanged. Models/features stay under ignored artifact directories. Older experiment model binaries/factor caches were removed to recover workspace space; their code, metrics, datasets and delivered predictions were retained. Use this current pipeline rather than the historical high-complexity ID-model scripts for the baseline.
