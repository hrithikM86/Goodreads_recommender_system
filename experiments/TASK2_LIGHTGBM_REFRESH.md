# Task2 LightGBM refresh

Original16 with-zero cached model scores/signals only; Mult-VAE excluded. Pair differences, sums and absolute differences; mirrored examples remain within the same split. One LightGBM configuration:31 leaves, min_child_samples150, learning rate.025, L2=30, L1=.1, max_bin127, sampling.9, seed42, two threads. Inner validation selected940 rounds and55% new/45% accepted margins before final evaluation. Accepted reference is XGBoost60%/original CatBoost40%, public accuracy.9513. Reference inner scores come from inner-trained models, not selection-wide fits.

Local evaluation: accepted.9477240742, LightGBM.9477133311, blend.9477777897. Five additional correct pairs out of93083: essentially flat locally. Historical cohorts reused; synthetic negative distribution may differ from leaderboard. Production meta model fits all186165 known holdout pairs; full-data component features reused. No base models retrained.

Checks: production model gzip save/reload prediction replay; candidate reversal antisymmetry; accepted prediction replay; original accepted CSV unchanged; both outputs20000 rows, preserved input columns/order, finite margins and valid candidate choices. Public scores pending.

Outputs: /workspace/output/task2_lightgbm_primary/task_2_predictions.csv and /workspace/output/task2_lightgbm_blend/task_2_predictions.csv.
