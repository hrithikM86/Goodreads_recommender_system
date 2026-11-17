# Task 6 quantile features

Adds 10 features from earlier-only 25th, 50th and 75th percentile predictions: log quantiles, interval width, left/right widths, relative width, asymmetry, anchor disagreement and crossing indicator. Existing missing feature values are handled by LightGBM.

Q25/Q75 use the accepted raw median settings (1171 trees, 31 leaves, minimum leaf 200, L2 50, learning rate .035). Median residual experts retain their selected settings: 15 leaves /1525 trees and31 leaves /1263 trees. No additional parameter search.

Both quantile models fit earlier targets only for historical meta features. Correction models fit penultimate targets; weights use inner validation users only. The final-gap cohort evaluates the locked weights. Production quantiles refit all known targets; correction models retain historical out-of-sample quantile inputs. Cohorts were reused across earlier experiments, so this comparison is not an untouched generalization estimate.

Local final-gap MAE: accepted7,102,518.494; new7,086,794.079 seconds. Public accepted score8,408,503.876; new public result pending. Selected weights: accepted20.42%, augmented15-leaf30.29%, augmented31-leaf49.29%.

CSV: /workspace/output/task6_interval_features/task_6_predictions.csv. Existing accepted CSV is unchanged.
