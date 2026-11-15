# Task 3: temporal rating ensemble optimization

Previous accepted Gradescope MSE: **0.704589533460822**. Its local evaluation
MSE is **0.6958509984**. Exclude all **23,888 rating-zero training rows**, keeping
**1,405,354 ratings in [1,5]**. Both output files still predict all 20,000 supplied
test rows; hidden ground-truth-zero rows cannot be identified from test inputs.

## New methods

1. Actual library SVD++ (`surprise.SVDpp`): 24 factors, 25 epochs, learning rate
   .007, regularization .08, seed 42, `cache_ratings=True`. Effective user vectors
   include the normalized sum of implicit book-history embeddings. Vectorized
   scoring is checked against the library's `predict` method. Additional components
   smooth its training residuals around user dates (30 days, shrinkage 3) and book
   dates (90 days, shrinkage 20).
2. Residual matrix factorization on top of the existing regularized temporal-bias
   model. Static rank 32 checkpoints at 30/60 epochs use regularization .06 and
   learning rate .01. Temporal rank 32 and rank 64 variants add user drift vectors
   and 180-day user-period vectors. Their score is
   `bias(u,i,t) + (p[u] + dev(u,t)*a[u] + c[u,period]*s[u,period]) dot q[i]`,
   where `c = sqrt(period_count/(period_count+10))`. User drift follows the lecture's
   signed time-deviation exponent .4, capped at +/-5. Temporal variants use
   factor regularization .08/.1, drift regularization .3, period regularization
   .2, learning rates .008/.007 and checkpoints 30/60 or 30/50. Rates decay by .98
   per epoch. The bias model is fitted first and remains fixed during residual MF;
   this is not a full jointly fitted TimeSVD++ implementation.
3. User-based neighborhoods complement the existing item-based ones. Binary
   overlap and positive Pearson-style centered-rating similarities are shrunk by
   `common_books/(common_books+10)`. Top 20/50 neighbors supply residual predictions
   with optional 365-day date weighting, plus support/confidence features. Self
   neighbors are excluded.
4. Local rating context: user/book rating means, variability, counts and residual
   means on the same UTC day and within 1/7/30/90/365 days; nearest known ratings
   before/after the query; rating-distribution fractions and entropy. All of these
   statistics come exclusively from fitted training rows.

The custom C++ MF kernel is checked against analytic SGD gradients and direct
prediction, including unknown-ID fallback. Temporal keys use 64-bit integers.
The temporal-bias component exactly matches the earlier cached predictions.

## Evaluation

Retain the previous seed 42 positive-rating split: 1,264,818 component-training,
70,268 selection and 70,268 evaluation ratings. Base models and history statistics
never see selection/evaluation ratings. Dates are query inputs, as in the task.
Known training events on either side of a query date may inform interpolation;
this is not next-interaction forecasting. Invalid dates outside 1970–2100 are
excluded from new date-conditioned statistics, but their positive ratings remain
in static model fitting.

The previous stacker is refitted inside selection's seed 17 inner 80/20 split
before blending, so it supplies no in-sample predictions to blend selection.
CatBoost depths 4/6/8, stopping iteration and new-model weights .5/.75/1 are
selected by inner tuning MSE. The final evaluation split is not fitted or used
for selecting the primary model. All component models then refit on all positive
training ratings; combiners refit on all selection ratings.

| Candidate | Inner tuning MSE | Final local evaluation MSE |
| --- | ---: | ---: |
| Previous model | 0.699877 | 0.695851 |
| Primary expanded ensemble | **0.689604** | **0.687612** |
| 50/50 blend with previous model | 0.691841 | 0.688954 |

The primary selected depth 8, 530 iterations and pure new-ensemble weight 1.
The blend is an independently labeled conservative upload candidate; final
evaluation scores did not select the primary. These are local results, not
leaderboard scores. A public upload is needed to measure the actual improvement.
User-based Pearson neighborhoods and residual MF have useful importance in the
new stacker; importance is not a causal component ablation.

## Audits and files

Both CSVs have 20,000 finite ratings in [1,5], with original headers, user/book
IDs, dates and row order preserved. Saved ensembles reproduce their predictions.
The old full-data model reproduces the accepted previous prediction file to
floating-point precision (maximum difference 8.9e-16). Independent calculations
verify sampled MF checkpoint predictions and all bare SVD++ predictions from
saved vectors. Held-out target user/book pairs are absent from the validation
training matrix. Data bytes match the latest supplied GitHub files.

Complete metrics, prediction/data/source hashes, feature importances and checks
are stored in `task3_optimization_results.json`. Fitted weights and feature caches
remain outside Git. The original Task 3 pipeline remains available for comparison.
The notebook now runs this candidate; its Task 4 cell retains the confirmed 50/50
blend at 0.9466, and Task 2 retains its with-zero model at 0.9505.

## Reproduce

```
pip install numpy pandas scipy scikit-surprise catboost
python experiments/task3_optimize_checks.py
python experiments/task3_optimize.py --data-dir 'Assignment 1/assignment1_2026' --output-dir experiments/artifacts/task3_optimized --predict
```

Requires `g++`. The runner generates the original baseline cache if needed. To
reuse an existing audited cache, pass `--baseline-dir PATH_TO_CACHE`; a legacy
Task 3 output root is also accepted. Use a fresh output directory after changing
source/settings. Model and feature caches are separated into validation/full
stages, input files are fingerprinted, and legacy validation labels/dates/order
are verified before reuse. The MF and SVD++ fitting workers run independently;
full fitting also regenerates baseline component features. The two exported
prediction candidates differ only in their final blend with the previous model.

## Accepted public result

The primary ensemble scored 0.6940055616901505 on Gradescope, versus
0.6964250044227708 for its 50/50 blend. The primary is the accepted reference
for the next optimization round.
