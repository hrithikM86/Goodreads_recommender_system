# Task 4: XGBoost and LightGBM instead of CatBoost

Accepted public accuracy entering this comparison: 0.9478, user-reported rank66. The accepted model is the complete component refit plus temporal FM with one CatBoost combiner. Its saved local evaluation accuracy is0.9388494745532653.

This experiment keeps the identical fitted component score bank: BPR32/64/128; ALS64/128 and separately configured ALS128; 180/365-day user-period ALS; offset365-day user-period BPR; EASE; SLIM; cosine/Jaccard neighborhoods; calendar density and activity; time-local history; reader drift/calendar-book bias; and temporal FM checkpoints5/15/30. Rating-zero events remain observed interactions. Only the final combiner changes; component models are not refitted in this comparison. No new data or external book metadata is used.

For each candidate pair, the combiner receives the difference and sum of each component score. Training mirrors candidate order and reverses its binary label. Predictions use P(item1 wins)-P(item2 wins), ensuring candidate-order antisymmetry. Training rows remain entirely separate from historical evaluation rows.

Selection uses the established seed17 80/20 split within71462 selection pairs, split before mirroring. XGBoost compares depth4/6/8 with minimum child weights20/30/50, respectively. LightGBM compares15/31/63 leaves with minimum child samples100/150/250. Both use learning rate.035, L2 leaf penalty20, row and feature sampling.9, maximum1600 iterations, seed619 and2 CPU threads. Logloss early stopping uses100 rounds. Within the resulting run, candidate-order accuracy on inner validation selects the tree count from100-tree checkpoints plus the stopping iteration. The strongest inner-validation configuration is refitted on all selection pairs, then evaluated on the untouched evaluation rows. Evaluation labels do not tune configuration or tree count.

These local holdouts have been reused over several experiment rounds and are not unbiased estimates of final hidden-test performance. Public-test scores of the new candidates are unknown until the user submits them.

XGBoost CPU3.4.1 was installed without changing existing packages; LightGBM4.7.0 was already available. Outputs retain all20000 input rows, supplied timestamps and candidate IDs. Each chosen book must equal one of the supplied candidates. Score finiteness, candidate-order antisymmetry and saved model replay are checked.

Reproduce with the existing saved component bank:

```
python experiments/task4_tree_combiners.py --kind xgboost
python experiments/task4_tree_combiners.py --kind lightgbm
```

The script prepends the local CPU XGBoost and LightGBM package paths for this environment. Full component refit sources are in task4_two_approaches.py and TASK4_TWO_FM_APPROACHES.md. Reproducible duplicate intermediate arrays were removed to conserve disk; accepted models, merged score banks and all prediction files are retained.

## Bounded initial comparison

The original three-configuration searches were bounded after completing XGBoost depth4 and LightGBM15/31-leaf trials. The unfinished deeper trials were stopped before evaluation and do not affect model selection. Completed XGBoost depth4 selected1553 trees with inner accuracy0.9416497586231022. Completed LightGBM15/31-leaf trials tied at0.9415098299867067; the15-leaf model is retained, with1100 trees. The accepted CatBoost inner accuracy was0.9412999370321137. The `--finish-completed` option reads completed trial records from the local training logs and refits their selected configuration; evaluation results remain excluded from selection. Final metadata records exactly which trials completed.

## Results

XGBoost local evaluation accuracy:0.9391713194240376; LightGBM:0.9386675622350027; accepted CatBoost reference:0.9388494745532653. XGBoost changes223 and LightGBM changes228 of20000 public/hidden test predictions compared with the accepted file. Public-test scores are unknown. Serialized-model replay and all output alignment checks passed for both candidates.
