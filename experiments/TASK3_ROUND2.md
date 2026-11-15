# Task 3: dataset-specific rating models, round 2

Accepted public reference: **0.6940055616901505 MSE**. Its prior 50/50 blend
scored 0.6964250044227708. The user reports a rank-50 target of **0.6750716**.
These new files require a leaderboard upload; local results do not establish
that the rank target has been reached.

All models use only the supplied Task 3/4 training file. The 23,888 zero ratings
are excluded before every fit, history, graph and aggregate, leaving 1,405,354
ratings in [1,5]. Both outputs predict every supplied test row.

## What changed

**Learned, signed rating neighborhoods.** For each target book, choose up to 64
or 256 neighboring books using reading overlap and residual-rating association.
Fit ridge regressions only on users with an observed target-book rating:

`w_i = argmin_w ||r_observed_i - A_i w||² + lambda ||w||²`.

Predictors are either the user's known rating residuals or binary reading
history. The target book is excluded from its own predictors. Residuals are
relative to the previous regularized temporal-bias model. Penalties 5/25/100
are tested for rating predictors and 25/100 for presence predictors. Static and
365-day absolute-date-weighted predictions are supplied to the stacker. Signed
weights can express both agreement and disagreement between books. The wider
models use a mathematically equivalent dual solve when there are fewer observed
readers than candidate predictors. All saved coefficient matrices have a zero
diagonal. This is observed-rating-masked ridge, not implicit SLIM or full EASE.

**Joint temporal SVD++ variants.** Unlike the previous residual MF with a fixed
bias model, these jointly learn user/item biases, user-day and user/book-period
biases, monthly effects, user rating drift, user/item factors, drift factors,
and implicit-history book embeddings. A user-block implementation caches the
implicit profile and applies the exact aggregate equivalent of per-rating
updates. A naive-update numerical test verifies the equivalence. Weakly
regularized variants overfit; stronger regularization and earlier checkpoints
are included in the bank. These are temporal SVD++ variants, not an assertion of
an exact reimplementation of every term in Koren's published TimeSVD++ model.

Configs: rank 24/reg.025 and rank 48/reg.035 at 40/80 epochs; rank 24/reg.01 at 40/80;
rank 24/reg.12, rank 48/reg.08 and rank 16/reg.2 at 15/30/50. Learning rates range
.005–.008 and decay by .985 per epoch. User-day ridge penalties 10–30 and item
period penalties 10–60 are normalized by observed group counts in SGD.

**Full-data LightGBM rating learner.** It fits all component-training ratings,
rather than just the selection rows used by the stacker. Features include
24-dimensional genre/exposure vectors from a TF-IDF-weighted binary reading
matrix, user/book vector interactions, dates, and user/book/year rating
summaries. Own-book contributions are removed from training user profiles.
Rating summaries use five-fold out-of-fold encoding: no row's target, nor any
label in its encoding fold, enters its training summaries. This property is
unit-tested. Query summaries use only known training labels. The model has
31 leaves, max_bin=63, learning rate .06, min_child_samples=150, L2=15, and fixed
300/600-tree prediction checkpoints. It uses numeric summaries, not direct
high-cardinality ID memorization.

**Shared-book Pearson neighborhoods.** Compute correlations only on books both
users rated, using raw ratings or temporal-bias residuals, instead of dividing
by whole-history norms. Require at least 3 common books and shrink by
`common_count/(common_count+20)`. Positive and signed/anti-correlated variants,
top 20/50 neighbors, and static/365-day predictions are tested. Candidate profiles
come from up to 1000 normalized-overlap and 500 high-overlap users; profiles unable
to contribute to any query are pruned. Correlations are exact within that
candidate pool, not an exhaustive all-user search. A C++ intersection kernel is
checked against NumPy, including negative correlation and insufficient overlap.
Ties are resolved stably by user order.

## Selection and results

Reuse the accepted seed 42 90/5/5 positive-rating split exactly: 1,264,818 base
training, 70,268 selection and 70,268 evaluation ratings. An inner seed 17 80/20
split of selection tunes CatBoost depth 6/8, stopping iteration and blend weight
with the accepted reference. Component models never see selection/evaluation
ratings. Both earlier and later known training dates may inform interpolation.
This is date-conditioned rating prediction, not next-event forecasting.

The first search compared all signals, learned neighborhoods, and neighborhoods
plus the large-data learner. The all-signal bank won internally. The final
comparison therefore fits two all-signal stacks: without and with the new
shared-book Pearson features. It chooses the primary by inner tuning MSE.
A fixed 50/50 average of these TWO NEW ensembles is the second upload candidate;
it is not a blend with the old 0.69401 submission. Final evaluation MSE is reported
independently and did not tune the averaging weight.

| Model | Inner tuning MSE | Local evaluation MSE |
| --- | ---: | ---: |
| Accepted reference | 0.689604 | 0.687612 |
| New bank without common-book Pearson | 0.683810 | 0.680028 |
| New bank with common-book Pearson (primary) | **0.682999** | 0.680089 |
| Fixed 50/50 blend of the two new banks | Not tuned | **0.679601** |

The primary uses depth 8 / 1335 trees; the other bank uses depth 6 / 1526. Both have
learning rate .025, L2=15 and seed 42. Component standalone tuning improvements
include a date-weighted learned rating neighborhood at 0.705759, versus roughly
0.7176 for the earlier item-neighborhood component. No single-component
leaderboard contribution has been measured.

## Verification

- Both files contain 20,000 finite ratings in [1,5], retaining headers, user/book
  IDs, timestamps and row order.
- Both serialized ensembles replay their exported predictions; their reference
  models reproduce the accepted 0.69401 prediction file.
- Serialized LightGBM checkpoint predictions, signed sparse linear scores and
  date-weighted scores are independently recalculated.
- Common-book Pearson predictions are independently checked without the C++
  kernel on sampled full-data queries.
- Validation target pairs are absent from the fitted rating matrix.
- Data hashes and explicit validation/full cache-stage manifests prevent mixing
  feature caches from different splits or input files.

Metrics, source/data/prediction hashes and verification results are stored in
`task3_round2_results.json`. Fitted models, caches and CSVs stay outside Git.
The original accepted model and its CSV are preserved.

## Reproduce

```
pip install numpy pandas scipy scikit-surprise catboost lightgbm
python experiments/task3_round2_checks.py
python experiments/task3_round2.py --data-dir 'Assignment 1/assignment1_2026' --output-dir experiments/artifacts/task3_round2 --prior-dir experiments/artifacts/task3_optimized --baseline-dir experiments/artifacts/task3 --predict
```

Requires `g++`. Missing prior caches are generated by the earlier pipeline.
Models for the two final banks live under `models/plain` and `models/common`;
shared component caches live under `validation` and `full`. A data/config/schema
check controls reuse of trained stackers. Use fresh output directories when
changing component hyperparameters or source. The notebook invokes this runner.
Tasks 1/2/4/5/6 retain their accepted implementations.
