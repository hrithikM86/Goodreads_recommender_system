# Task6: median LightGBM experts and learned ensemble

Accepted public reference:8,452,714.299367478 seconds MAE. Preserve its prediction CSV. This experiment uses only the supplied Task6 data and the accepted training-only feature banks: history/gaps/rating summaries, book dates conditional on the reader's last date, and leave-query-user-out peer durations. The supplied next-book ID is an input. No hidden labels, external data or cross-task matching are used.

## Bounded comparison

LightGBM quantile loss with alpha0.5 targets the conditional median. Existing MAE models already target this quantity; the experiment tests different capacity/regularization and complementary errors, not a theoretically new optimum from renaming the loss.

Four fixed initial configurations (seed3139, max_bin127, L1 regularization0.1, path_smooth10, row/feature sampling.9, two CPU threads):

| Expert | Target | Leaves | Min leaf | L2 | Rate | Selected trees |
| --- | --- | --- | --- | --- | --- | --- |
| Raw median | gap in days |31|200|50|.035|1171|
| Log median | log1p gap |63|200|50|.035|244|
| Correction15 | gap minus older accepted expert forecast |15|150|30|.025|1525|
| Correction31 | same residual |31|200|50|.025|1263|

Direct experts fit192034 earlier historical targets. Early stopping uses80% of55403 penultimate targets, with150-round patience and max2000trees. The remaining20% of penultimate targets selects the ensemble. Correction experts fit the80% penultimate subset, using the20% for stopping and comparison. Their inputs add the older accepted expert forecast to the132 history/book/peer features. The current CatBoost calibration's inner model supplies a matching held-out accepted reference.

Fit nonnegative, sum-one ensemble weights by minimizing MAE on the inner20% only, including the accepted CatBoost calibration and all four new candidates. Use equal and vertex starts in SLSQP. Lock settings/weights before inspecting the final-gap evaluation. Readers' earlier targets can appear in fitting; their final targets remain excluded. Historical cohorts have been reused across experiments and are diagnostics rather than untouched hidden tests.

Selected weights:
- Accepted final CatBoost calibration:0.
- Raw median:0.008621097809580928.
- Log median:0.
- Correction15:0.44638134950997876.
- Correction31:0.5449975526804404.

The older accepted lower-level forecast remains the anchor/input of both correction models; a zero weight on the final CatBoost calibration does not remove those lower-level experts.

## Separate final-gap comparison

| Predictor | MAE seconds |
| --- | --- |
| Accepted final calibration |7,155,338.8505|
| Raw median |7,564,793.9891|
| Log median |7,654,302.2142|
| Correction15 |7,107,349.2631|
| Correction31 |7,099,997.9428|
| Inner-selected ensemble |7,102,518.4941|

The ensemble weights remain those selected on inner validation even though Correction31 alone happens to score slightly better on this final holdout. Paired improvement52,820.36seconds (about0.74%); approximate user-IID95% interval for new-minus-accepted MAE[-60,385.24,-45,255.47]seconds. This conditional interval ignores shared-book/model dependence and repeated experimental selection. It is not a public-score or rank guarantee.

## Production

Refit selected direct experts on all302840 known historical targets using full fitting-feature banks. Refit correction models on both known penultimate/final targets (110806 rows), whose lower-level forecasts were out of sample. Score unlabeled test inputs using all provided training history and the existing full-data lower-level forecast. Export finite nonnegative seconds; retain all55403 test rows/IDs/order/header. No independent local score is claimed for the final all-target production weights.

```
PYTHONPATH=/workspace/work/task3_lgb:/workspace/work/task2_packages python experiments/task6_quantile.py --stage experiment
PYTHONPATH=/workspace/work/task3_lgb:/workspace/work/task2_packages python experiments/task6_quantile.py --stage produce
PYTHONPATH=/workspace/work/task3_lgb:/workspace/work/task2_packages python experiments/task6_quantile_checks.py
```

Requires the accepted Task6 cached feature/model banks. New model files, margins and prediction CSVs remain local. Only sources/results go to codex. Use a new output directory if data or configurations change. Public/hidden results of this candidate are unknown.

The exported CSV contains55403 rows and passes saved-model replay, finite/nonnegative-second, exact evaluation-target alignment, and test-ID/order/header checks. Output SHA256:c94080bbf42c81d99c5c342b038dfeaebd5bc30f2d2711c745e8f35e2b5309f9. Mean absolute prediction change versus accepted488623.06seconds.
