# Task 4: temporal ensemble optimization

The accepted previous submission scored **0.9403** on Gradescope, with local
held-out accuracy **0.9327764**. All 1,429,242 supplied interactions are retained,
including the 23,888 rating-zero rows. Rating magnitudes are otherwise unused.
Zero-rating removal applies to rating Tasks 1 and 3, not this interaction task.

## New signals

- Static BPR-128: 250 iterations, learning rate .03, regularization .03, seed 42.
- ALS-128: 30 iterations, alpha 40, regularization .1, seed 42. Its normalized book
  vectors also supply similarities to the user's date-local history.
- Two temporal BPR-64 models: one user vector per (user,365-day period), with
  offsets 0 and 182 days; 150 iterations, learning rate .03, regularization .03.
  A missing period falls back to a complete static BPR score, not a mixture of
  vectors from different latent coordinate systems. Period availability and
  interaction count tell the combiner how much local history is available.
- EASE on the 4,000 most frequent fitted books, penalties 100/500/2000. This is a
  bounded-vocabulary approximation, covering about 79.3% of supplied test
  candidates. It does not claim full-vocabulary dense EASE.
- Nonnegative, 32-neighbor restricted SLIM at L1=.2 and 2, L2=10, using the same
  numerical solver checked against ElasticNet in Task 2.
- Time-weighted SLIM/EASE and latent-neighborhood scores, using
  `exp(-abs(history_date-query_date)/scale)` at 30/180/365 days, plus preceding
  and following known history neighbors and context density.
- Book-date counts, fractions and global-exposure-adjusted lifts at 1/7/14/60/
  180/730 days; position within a book's observed activity, robust date deviation,
  distances to previous/next observed book events, and exact two-sided exponential
  kernel densities at 7/30/90/365 days. Density lift divides by the global training
  date density to account for changes in overall platform activity.

The C++ density kernel evaluates the exact sum of exponentials through sorted
forward/backward recurrences. It avoids constructing a query-by-event matrix.
Tests compare it against direct summation and verify translation invariance.
Temporal history scores are checked against a hand-computed example.

These features extend the temporal lecture's time-varying latent preferences,
sliding windows and exponential history weighting. They are not a full TimeSVD++
or FPMC implementation. Existing static/temporal ALS, BPR and similarity features
remain available to the new combiner.

## Validation and leakage controls

Reuse the accepted seed 42 row split exactly: 1,286,317 component-training rows,
71,462 selection positives, and 71,463 evaluation positives. Reconstruct the same
seed 43 sequential popularity-weighted background draws, allowing already-seen
candidates. Cached baseline query user/item arrays must match exactly. Baseline
and new caches are isolated by data manifests and validation/full-fit directories.
Use a fresh output directory when changing model settings or source code.

Every fitted factor, book-date statistic, history and linear coefficient uses
only the component-training partition during validation. Held-out target dates
are query inputs, as dates are supplied for Task 4; the target interactions are
not included in training metadata. Both earlier and later *known training*
interactions may inform the query. This is date-conditioned interaction
prediction/interpolation, not the next-interaction forecasting protocol of Task 5.
Invalid dates outside 1970–2100 remain static interactions but do not contribute
to new date statistics.

An inner 80/20 split of selection pairs chooses CatBoost depth 5/7/9, stopping
iteration and new/previous-model blend weight .25/.5/.75/1. The previous
architecture is refitted inside this inner split before blending, avoiding
in-sample predictions from a combiner fitted on all selection labels.
Mirrored candidate orders and explicit prediction symmetrization enforce score
antisymmetry. The final evaluation split is excluded from fitting and selection.
Selected combiners refit on all selection pairs; component models then refit on
all supplied training interactions for test prediction. No test answers are used.

## Reproduce

```
pip install implicit numpy pandas scipy scikit-learn catboost
python experiments/task4_optimize_checks.py
python experiments/task4_optimize.py --data-dir 'Assignment 1/assignment1_2026' --output-dir experiments/artifacts/task4_optimized --predict
```

Requires a C++ compiler (`g++`). With no existing baseline manifest, the runner
first generates the previous Task 4 feature cache. To reuse an audited cache,
pass `--baseline-dir experiments/artifacts/task4`. Input fingerprints, exact query
arrays, original row split and candidate order are verified before reusing it.
New factor fits run two independent single-threaded jobs concurrently; stacking
uses two threads. Generated predictions, factors and caches stay outside Git.
The original Task 4 runner is preserved for comparison.

## Results and upload candidates

| Model | Inner tuning accuracy | Final local evaluation |
| --- | ---: | ---: |
| Previous accepted architecture | 93.3324% | 93.2776% |
| Selected expanded ensemble | 94.1580% | **93.8961%** |
| 50/50 blend of expanded and previous models | 93.8921% | 93.7548% |

Depth 5 / 888 iterations won over depths 7 and 9. Selection chose the pure
expanded ensemble. The 50/50 blend is a separately labeled conservative upload
candidate, already evaluated in the same inner grid; final evaluation scores did
not choose the primary model. These are local metrics, not leaderboard results.

All 20,000 rows in each output passed header, ID, timestamp, candidate-membership
and order checks. Saved models reproduce all choices exactly. The refitted
previous architecture reproduces every accepted previous test choice. Sampled
static SLIM and date-weighted SLIM/EASE scores independently match direct sparse
matrix/history calculations. Held-out positive user/book pairs are absent from
the fitted validation matrix. All restricted SLIM fits met the convergence
criterion. The candidates differ on 228 rows; they change 490 and 262 choices,
respectively, from the accepted previous submission.

`task4_optimization_results.json` records metrics, source/data/prediction hashes,
component scores, feature importances and audit results. The notebook Task 4 cell
runs the selected pipeline. Its Task 2 cell now points to the confirmed accepted
with-zero model (95.05% public accuracy); Tasks 1, 3, 5 and 6 remain unchanged.
