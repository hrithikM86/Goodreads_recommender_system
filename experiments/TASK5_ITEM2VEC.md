# Task5: one new Item2Vec model added to accepted neural architecture

Public reference: accepted neural ensemble0.9219689811720785. Recent standalone boosting candidates scored0.92058627 (CatBoost),0.92001014 (XGBoost),0.92012537 (LightGBM), so none replaces the reference.

## Addition

Train one32-dimensional skip-gram Item2vec model with negative sampling on ordered book-ID histories. Use12 epochs, dynamic window up to5, three negative samples, frequency^.75 noise sampling through a524288-entry table, minimum count2, seed1777. Initialization and SGD come from the shared native kernel in task4_item2vec.cpp; no Task4 data or weights are used. Only user/item columns and CSV order are read. Numeric timestamps and rating values are not features. Zero-rated reading events are retained; rare books are filtered independent of rating.

Validation embeddings fit only each reader's H[:-2] fitting prefix. Evaluation context H[:-1] includes the legitimately known penultimate event, while its hidden final target is excluded. Full-test embeddings fit all provided training history. No test labels are available. Books without trained contextual support receive zero vectors/scores.

Eight candidate signals are appended to each accepted neural network's existing input bank:

- Input-vector cosine with last book, recent3 mean, recent10 mean, .9-decay last20 mean, last20 maximum.
- Input/output SGNS compatibility with last book, recent3 mean and .9-decay last20 mean.

Only trained context vectors contribute. A candidate already in the known history is excluded from pooling to avoid trivial self-cosine. Recency is positional, not based on dates. Cold/unknown candidates receive zero Item2vec scores and remain covered by the previous component models.

## Controlled neural refit

Keep the accepted architecture/settings and its existing50/50 final combination:

- Sequence network: original224 scores +8 Item2vec scores; hidden64/32, seed71,16 epochs, alpha.1.
- Preference network: original260 scores +8 Item2vec scores; hidden64/32, seed73,8 epochs, alpha.1.

Both take differences/sums, fit their StandardScaler on mirrored86786 selection pairs only, and train with Adam through sklearn MLPClassifier. The old components are reused. No new hand-assigned percentage is applied to Item2vec; its input weights are learned. No new hyperparameters, checkpoints or blend weights are selected from evaluation labels in this comparison. Evaluate on86786 final-event pairs, exactly as before. These cohorts have been reused in earlier experiments and are diagnostics rather than unbiased hidden-test estimates.

## Result and checks

Local accuracy improved from0.9235245316064803 to0.9248841979121056. The candidate changes1307 of86786 test decisions. Public score remains unknown; retain the accepted file unless it improves.

Independent dense formulas check all eight position-based similarity/SGNS scores, self exclusion, cold and unknown handling. Cached sampled factors/context vectors replay128 validation and128 test rows. Every historical target is absent from its own history, and prefix-only training-event counts match. Serialized networks replay every final output; candidate-swap antisymmetry, all86786 input IDs/rows, valid candidate choices and accepted-reference replay pass.

## Reproduction and storage

```
mkdir -p /workspace/work/task5_item2vec
g++ -O3 -shared -fPIC experiments/task4_item2vec.cpp -o /workspace/work/task5_item2vec/item2vec.so
g++ -O3 -shared -fPIC experiments/task5_item2vec.cpp -o /workspace/work/task5_item2vec/score.so
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_item2vec.py --stage validation
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_item2vec.py --stage full
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_item2vec.py --stage combine
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_item2vec_checks.py --replay
```

Requires the existing accepted component-score banks and local Python packages. Full Item2vec tables over232427 books would occupy about60MB and are not retained on this nearly full workspace. Training regenerates them deterministically in memory; query scores and sampled embedding/context tables are retained. Both complete neural models/scalers and all prediction margins are retained. Cached features are specific to this dataset/cohort; use a fresh output directory if inputs change. Rejected boosting and unaccepted experimental reader-score caches were removed to make room, recorded in task5_item2vec_cleanup.json; all accepted models/banks, original datasets, sources/results, margins and prediction CSVs are preserved. Only code/results are published to codex.
