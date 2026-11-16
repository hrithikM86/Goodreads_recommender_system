# Task5: fresh standalone boosting over the complete accepted component bank

Goal: improve the accepted0.9219689811720785 public accuracy (latest reported rank126) toward rank below100. Tasks2/4 were last reported at96/60. These are user goals; the exact rank-to-mark formula has not been provided.

## Accepted method

The accepted ensemble averages the scores of two neural pair rankers: a sequence/order/attention ranker and a preference-augmented ranker. Its historical evaluation accuracy is0.9235245316064803. Raw component scores include BPR/ALS preferences; FPMC32/64 variants; PRME-style and TransRec metric models; truncated FOSSIL; EASE; weighted item/reader/session neighborhoods; popularity/seen/cold indicators; skip-order graphs and two-hop paths; order-inferred era factors; candidate-aware attention; stronger BPR128/BPR96; and query-user factor adaptation.

Numeric rating values and timestamps are not model features for Task5. All reading events, including rating-zero events, remain observations. CSV order determines each reader's history. Inferred order/era features are derived from that ordering, not from timestamp values.

## New comparison

Reuse all260 candidate component scores from the accepted bank. Their differences/sums produce520 pair features. Replace the final neural-combiner blend with one fresh standalone classifier per candidate solution:

| Candidate | Initial fixed settings |
|---|---|
| CatBoost |800 trees, depth6, learning rate.05, L2 leaf penalty20, seed1429 |
| XGBoost CPU |1000 trees, depth5, minimum child weight30, learning rate.04, L2=30, row fraction.9, feature fraction.8, seed1429 |
| LightGBM |1000 trees,31 leaves, minimum200 leaf samples, learning rate.04, L2=30, L1=.1, row fraction.9, feature fraction.8, seed1429 |

These are fixed initial configurations, not a claimed exhaustive tuning search. Evaluation outcomes do not choose hyperparameters or stopping counts. Each file is a complete standalone boosting solution with no manually assigned old/new blend weights. The old final neural ensemble predictions are references for evaluation, not additional in-sample training features.

## Chronological validation

Reuse the established cohorts and negatives. Fit component models on H[:-2] for each of86786 readers. Selection positives are the penultimate event with H[:-2] context; evaluation positives are the final event with H[:-1] context. Evaluation context legitimately includes the known penultimate event; its target remains excluded. The existing data loader checks query IDs against the original baseline caches. Each fresh combiner fits86786 selection pairs, mirrored to173572 labeled rows. The86786 final-event pairs evaluate it. Test-time scores use the previously fitted full-training component bank and all known training history, without test labels.

Negative samples and cohorts remain identical to earlier experiments. Historical cohorts have been reused across many experiments; local results are diagnostics, not unbiased hidden-test estimates. Only user-reported leaderboard scores establish public performance. Preserve the accepted0.92197 file unless a candidate improves it.

## Audits and reproduction

Check exclusion of every selection/evaluation target from its own context. Replay the accepted test margins against its preserved CSV. For each new model, check all86786 input rows, IDs and candidate choices, finite scores, pair-swap antisymmetry, serialized-model replay and dataset hashes. Query feature names must match between validation/test and contain all260 columns.

```
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_boosting.py --kind catboost
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_boosting.py --kind xgboost
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_boosting.py --kind lightgbm
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_boosting_checks.py --replay
```

The existing accepted-bank generation scripts are dependencies. XGBoost CPU/LightGBM paths are the local package directories used in earlier experiments. Models use two CPU threads each. Only source and result metadata are published to codex; datasets, models and CSVs remain local. Unaccepted/rejected combiner checkpoints were removed to make room, recorded in task5_boosting_cleanup.json; accepted models/banks, source/settings, prediction files and result margins are retained.

## Results

Historical evaluation accuracy: accepted neural ensemble0.9235245316064803; CatBoost0.9225912013458392; XGBoost0.923605189777153; LightGBM0.9235590993939115. The new files change1905,1838 and1899 predictions respectively versus the accepted CSV (exact counts in JSON). Both positive local differences are tiny, and CatBoost is weaker locally. No public improvement or below100 rank is claimed. All86786 rows per file, source hashes, feature schemas, saved-model replay and the accepted-reference replay passed. Public results await user submission.
