# Task4: LightGBM tuning, accepted bank frozen

Public reference is Item2vec-LightGBM accuracy0.9486, rank57. Rejected FISM/FPMC candidates scored0.9477/0.9479. Neither component is included in this experiment.

The exact accepted142-score candidate bank is unchanged: BPR, ALS, user-period models, SLIM/EASE, neighborhood/calendar/time-local history, reader drift, temporal FM, and Item2vec checkpoints. Candidate differences/sums give284 classifier features. Pin all six input score-array hashes before training. No underlying component model, embedding or feature is retrained/recomputed.

## Selection protocol

Use the established71462 selection pairs and seed17 80/20 inner split BEFORE pair mirroring. Compare three predeclared settings. All use binary logloss, seed619, two CPU threads, and row resampling every iteration.

| Trial | Leaves | Min leaf samples | L2 | L1 | Learning rate | Row fraction | Feature fraction | Maximum trees |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Accepted settings |15|100|20|0|.035|.9|.9|1100|
| Regularized15 |15|200|50|.1|.035|.85|.8|1600|
| Regularized31 |31|250|50|.1|.025|.9|.8|1600|

The baseline is fit for its known1100 trees without stopping. Alternatives use inner-validation logloss early stopping with100 rounds; candidate-order accuracy then selects the prefix count from200-tree checkpoints, the stopping iteration, and1100 if available. Candidate-order scores are `P(first wins)-P(swapped first wins)`. The greatest inner accuracy chooses the final configuration, retaining earlier configurations on exact ties; tree-count ties choose the shorter prefix. The selected model is refitted on all71462 selection pairs and checked once on71463 historical evaluation pairs. Evaluation labels never select parameters or counts. These historical cohorts have been reused across previous experiments, so local scores are diagnostics rather than unbiased hidden-test estimates.

Trial models and inner margins are retained for reproducibility. Resume uses cached completed trial metadata only if the source-bank hashes and trial settings agree.

## Inner results

- Accepted15-leaf setup:0.9408101868047296 at1100 trees.
- Regularized15:0.9411600083957182 at1200 trees.
- Regularized31:0.941229972713916 at1200 trees, selected.

The selected31-leaf model has minimum250 samples per leaf, L2=50, L1=.1, learning rate.025, row fraction.9 and feature fraction.8. This is one tuned LightGBM, not a blend of the trial models.

## Verification

Compare pinned bank hashes with the accepted experiments, and verify raw training/test CSV hashes. Assert142 candidate scores /284 classifier features. Replay the selected saved inner-model prefix against its cached inner margins and recorded accuracy. Replay the final serialized LightGBM on historical evaluation and all test pairs; verify all20000 CSV rows/timestamps/IDs/candidate decisions and the original accepted CSV checksum. Candidate-swap antisymmetry and finite margins are checked during export.

## Reproduction

```
PYTHONPATH=/workspace/work/task2_packages python experiments/task4_lightgbm_tune.py
PYTHONPATH=/workspace/work/task2_packages python experiments/task4_lightgbm_tune_checks.py
```

Requires the existing accepted component-score bank, Item2vec caches and local LightGBM installation from earlier experiments. Only code/results go to codex; datasets, factors and prediction CSVs remain local. Rejected FPMC parameter/feature cache files were removed for disk space, recorded in task4_lightgbm_tune_cleanup.json. Their code/settings, result metadata, margins and prediction files were retained. Accepted artifacts are unchanged.

## Final result

Selected31-leaf/1200-tree model: local evaluation accuracy0.9395771238263158 versus accepted0.9392132991897906. It changes131 of20000 predictions. Bank and raw-input hashes match the accepted experiment; all selected-inner/final model replay and output checks passed, and the accepted CSV is unchanged. Public accuracy remains unknown. Keep the0.9486 submission unless this scores better.
