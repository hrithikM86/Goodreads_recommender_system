# Task 3: low-compute LightGBM combiner

Reuses the accepted245-column positive-only out-of-sample component bank, adding24 features: valid-date/calendar encodings, log user/item support, rating dispersion, previous/next/nearest observed timestamp gaps, history percentile/extrapolation, and disagreement summaries of rating predictors. Validation query labels do not enter the base models or these aggregates. All validation and training ratings are positive.

One LightGBM configuration: 63 leaves, minimum200 rows/leaf, learning rate .025, L2=30, L1=.1, max_bin127, feature/row sampling .9, two threads. Early stopping chooses278 trees on the existing inner20% split. No XGBoost, CatBoost, or expensive base learners were trained. Existing accepted CatBoost predictions are reused for a blend.

Blend weight selected by bounded least squares on inner validation: LightGBM50.0994%, accepted49.9006%. Locked before final validation. Production fits the combiner on all140536 meta rows using already available full-data component predictions for the20000 test rows. Historical splits have been reused across prior experiments; these scores are not an untouched holdout. No leaderboard result is known for the new files.

Local evaluation MSE: accepted0.6791676212; LightGBM0.6785149594; blend0.6776485554.

Exports: /workspace/output/task3_lightgbm_primary/task_3_predictions.csv and /workspace/output/task3_lightgbm_blend/task_3_predictions.csv. Both preserve20000 test rows and original input columns; ratings finite in[1,5]. Saved production model replay verified. Accepted CSV unchanged.
