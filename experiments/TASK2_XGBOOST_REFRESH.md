# Task 2: low-compute XGBoost combiner

All16 cached scores/signals from the accepted with-zero solution are used: BPR64/128/96, ALS128, EASE50/300/1000, cosine max/top10, Jaccard max, SLIM02/2, popularity, already-seen status, user count and EASE coverage. No base models retrained. Pair design uses differences, sums and absolute differences; each pair is mirrored and the resulting probabilities antisymmetrized. Thus reversing candidates reverses the decision margin.

One histogram XGBoost configuration: depth5, min_child_weight50, learning rate.035, L2=20, L1=.1, max_bin127, row/column sampling.9, seed42, two CPU threads. Inner20% of the selection cohort chooses1041 trees via early stopping. The accepted CatBoost architecture is refitted on inner training only to obtain honest inner reference margins. A21-point weight grid on the same inner cohort chooses60% XGBoost/40% accepted. Settings and weights locked before evaluation.

Local evaluation accuracy: accepted.9467464521; XGBoost.9477025880; blend.9477240742. Production XGBoost fits all186165 held-out training pairs with existing synthetic negatives and full-data cached base scores for test. Existing accepted CatBoost remains unchanged. Historical cohorts are reused; the negative-sampling distribution may differ from the leaderboard. New public scores pending.

Checks include input hashes, component column ordering, nonoverlapping splits, accepted reference margin replay, production model replay, reversed-candidate antisymmetry,20000 output rows, preserved test IDs/order, finite margins, valid candidate choices and unchanged accepted CSV.

Outputs: /workspace/output/task2_xgboost_primary/task_2_predictions.csv and /workspace/output/task2_xgboost_blend/task_2_predictions.csv.
