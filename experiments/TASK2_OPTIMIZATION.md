# Task 2: zero-rating ablation and complementary models

This experiment produces two independent prediction files for the same supplied
20,000 candidate pairs: `with_zero` retains every observed interaction;
`without_zero` excludes rows with rating <= 0. Ratings otherwise do not enter
features. Timestamps are not used.

## Models

- BPR-64: 200 iterations, learning rate .03, regularization .01, seed 17.
- BPR-128: 350 iterations, learning rate .03, regularization .03, seed 42.
- BPR-96: 300 iterations, learning rate .05, regularization .003, seed 23.
- Implicit ALS-128: 30 iterations, alpha 40, regularization .1, seed 42.
- EASE on the 2,000 most frequent training books, ridge penalties 50/300/1000.
  Restricted vocabulary is an explicit memory approximation: uncovered candidates
  receive zero EASE score and a coverage feature. No full-vocabulary EASE claim.
- SLIM on each target's 32 highest-cosine co-occurring neighbors, nonnegative
  coefficients, no self edge. Minimize
  `.5 ||X_j - X_N b||² + 10/2 ||b||² + l1 ||b||_1`, with l1=.2 or 2.
  This is restricted-neighborhood SLIM, not unrestricted global SLIM.
- Maximum cosine, top10 cosine sum, maximum Jaccard, log item popularity,
  already-seen flag and log user history size.

Factor scores for users/items with no fitted interactions are set to zero.
A small C++ coordinate-descent kernel speeds up restricted SLIM; numerical checks
compare it to scikit-learn ElasticNet. The library compiles into ignored artifacts.

## Combination and validation

Same seed 42 random 90/5/5 row split as the accepted baseline. Zero filtering is
applied *after* the common split, so held-out rows never enter component training.
For no-zero, rating-zero selection/evaluation positives are also removed.
Background items are drawn proportionally to each variant's training popularity;
previously seen candidates are permitted. These are synthetic background labels,
not explicit negative user feedback. The variants share held-out positive row IDs;
their sampled background candidate IDs need not be identical.

Selection has a further fixed 80/20 split. It alone chooses logistic regularization,
old-style vs expanded linear features, CatBoost depth 3/5/7, stopping iteration,
and linear/nonlinear mixture weight. CatBoost receives feature differences and
sums, learns from mirrored candidate orders and is symmetrized at prediction:
`p(d,s) - p(-d,s)`. Every selected combiner is refitted on all selection examples.
The final 5% evaluation partition is excluded from training and model selection.
Both each variant's full evaluation and nonzero-positive evaluation are reported.

After selection, component models refit on all allowed training interactions;
combiner weights remain frozen. Supplied test IDs are used for scoring and fitting
the corresponding sparse SLIM columns; SLIM coefficients depend solely on training
interactions. No hidden labels or test outcomes are used in model selection.

## Run

```
pip install implicit catboost numpy pandas scipy scikit-learn
python experiments/task2_optimize_checks.py
python experiments/task2_optimize.py --data-dir 'Assignment 1/assignment1_2026' --output-dir experiments/artifacts/task2_optimized --variant with_zero --predict
python experiments/task2_optimize.py --data-dir 'Assignment 1/assignment1_2026' --output-dir experiments/artifacts/task2_optimized --variant without_zero --predict
```

Requires `g++`. Use a fresh output directory after changing model hyperparameters
or source code; factor caches are local to this experiment configuration.
The input manifest rejects changed data files in an existing experiment directory.
Original Task2 runner and Tasks 1/3/4/5/6 are unchanged. Results and prediction
validation summaries accompany the final experiment after fitting.

## Completed local validation

| Training variant | Evaluation positives | Linear combiner | Selected nonlinear combiner | Nonzero-positive evaluation |
| --- | ---: | ---: | ---: | ---: |
| Keep rating-zero interactions | 93,083 | 94.4791% | 94.6746% | 94.6985% |
| Remove rating-zero interactions | 91,729 | 94.6375% | 94.8152% | 94.8152% |

Both selected a pure CatBoost combiner, depth 7: 275 iterations with zeros and
478 without zeros. Its inputs still include all factor, neighborhood and
supporting scores; this is a stacked ensemble, not an ID-only CatBoost model.
All restricted SLIM regressions converged within 100 coordinate sweeps, with
maximum final coefficient change below 1e-5. Earlier accepted Task2 local accuracy
was 94.3405%, with reported public accuracy 94.61%. Its random background draws
were generated sequentially; this run draws them in batches, so the exact pairs
differ despite the same sampling distribution. Public upload is needed for a
fair comparison on identical test candidates and hidden labels.

The nonzero-positive evaluation rows are shared across variants, but their
background draws differ. The modest local advantage of removing zeros is not
proof of a leaderboard improvement. CatBoost feature importance shows substantial
use of SLIM and smaller use of restricted EASE; it is not a causal ablation.
The low-regularization BPR-96 candidate is weaker alone than BPR-64/128, but receives
some ensemble weight through learned tree splits. No component was selected by
looking at test answers.

## Prediction audit

Both outputs contain exactly 20,000 valid candidate choices with original IDs,
header and row order. Saved combiners reproduce every exported choice exactly.
Independent sparse-matrix multiplication reproduces sampled SLIM scores.
The variants disagree on 279 rows. All supplied training/test input bytes are
unchanged. Complete metrics, checksums and audit results are in
`task2_optimization_results.json`. Prediction CSVs and fitted artifacts remain
outside Git.
