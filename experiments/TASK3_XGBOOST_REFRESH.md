# Task 3: XGBoost combiner

Uses the same 269-column positive-only cached component bank and context/disagreement features as the LightGBM refresh. No base learners refitted. One histogram XGBoost configuration: depth5, min_child_weight100, learning_rate.03, L2=30, L1=.1, row/column sampling.9, max_bin127, seed42, two threads. Early stopping on inner validation chose720 trees.

Reference is the accepted LightGBM/CatBoost blend (public MSE0.6818167471381362). Its historical inner predictions are reconstructed from selection models, rather than full-data models. A bounded least-squares blend selects42.15098% XGBoost and57.84902% accepted predictions using inner validation only. Settings and weights locked before final validation. Historical splits were reused in earlier experiments; these are not untouched holdout scores.

Local final validation MSE: accepted0.6776485554; XGBoost alone0.6791048187; blend0.6774341252. Production combiner uses all140536 held-out training ratings and existing full-training-data base predictions for test inference. New public scores pending.

Exports: /workspace/output/task3_xgboost_primary/task_3_predictions.csv and /workspace/output/task3_xgboost_blend/task_3_predictions.csv. Checks enforce20000 rows, original test input columns/order, finite ratings in[1,5], input/cache hash consistency, saved production model prediction replay, and unchanged accepted CSV.
