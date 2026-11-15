# Tasks 1, 5 and 6: second improvement round

The latest user-reported accepted public scores and ranks are recorded in
`PUBLIC_SCORES.json`. They are 0.5385673542232966 MSE (Task 1),
0.9107920632360058 accuracy (Task 5) and 8,890,395.0326342 seconds MAE (Task 6).
These are public-half results, not hidden-test estimates. Previous accepted
forecasts remain available; this round exports separate candidates.

## Task 1

Retain all 23 features from the accepted SVD++/rating-neighborhood stacker.
Add user-side and wider item-side learned neighborhoods: up to 128 neighbors
from reading cosine and signed residual correlation, separate exposure/rating
penalties, rating/exposure penalty ratios 1 and 4, and ridge penalties 20/100.
Add signed Pearson summaries (40 neighbors, signed power 1/2).

Compare 64-factor lightly regularized SVD++ (.04, checkpoints 50/100/180),
96-factor strongly regularized SVD++ (.3, checkpoints 30/70/140), and
**item-side SVD++** (32 factors, .1, checkpoints 20/50/100). Item-side SVD++
swaps readers and books, modeling a book through its known readers. It uses
exactly the same cached implicit-update kernel previously checked against naive
updates. Unseen factors are neutral. Bigger standalone factor models did not
outperform the accepted ensemble; their complementary signals are evaluated by
the stacker rather than presumed better.

The positive-only seed-42 component split remains 90/5/5. The inner seed-17
80/20 split of selection examples chooses CatBoost depth 4/6, iterations, depth
mixture and shrinkage toward the accepted reference. Separate evaluation
labels do not fit these choices. For export, components fit all positive rows
and meta-models refit the available held-out component forecasts and labels.

## Task 5

The temporal lecture discusses FPMC, Personalized Ranking Metric Embedding and
translation-based recommendation. Add two 32-dimensional pairwise-BPR models:

- PRME-style score: `bias[i] - .5||P[u]-Q[i]||² - .5||Q[last]-Q[i]||²`.
  This is a generic metric variant, without POI geographical information.
- TransRec-style score: `bias[i] - ||Q[last]+P[u]+global-Q[i]||²`.

Train with unit-ball projections, user-uniform transition sampling, mixed
popularity/uniform negatives and known-history rejection; checkpoints 20/60/120.
Cold previous-book transitions are neutralized by the score loader. Preserve
personal preference and candidate bias as separate features.

Add a truncated FOSSIL variant: `bias[i] + (sum(A[history20])/sqrt(n) +
sum_l(global_weight[l]+user_weight[u,l])*A[last_l]) dot B[i]`, using the most
recent 20 books and three personalized Markov coefficients. It jointly learns
shared history/target embeddings, biases and transition weights with BPR.
24 factors, learning rate .015, regularization .03, checkpoints 5/15/35.
This is a truncated high-order model, not a full-history FOSSIL reproduction.

Add EASE on the 4,000 most frequent fitted-prefix books, ridge penalties
50/300/1000. Its score uses whole, latest, latest-three and .9-decayed contexts.
EASE solves a zero-diagonal regularized linear autoencoder exactly. Rare books
retain the existing model signals.

Retain the entire accepted sequence-model feature bank. New classifiers use
pair differences/sums, mirrored training pairs, and antisymmetric margins.
The existing chronological two-event holdout and seed-43 negative candidates
are unchanged. Tune depth 5/7, checkpoints, mixtures and reference shrinkage on
the same inner user split as before. All ratings are retained; only ordering,
user IDs and book IDs are used. No numeric timestamp or rating value is used.

## Task 6

Add leave-query-user-out peer duration summaries for the supplied next book:
log-gap quantiles, spread, zero-gap rate, pace-normalized quantiles, calendar-
weighted and reader-pace-weighted summaries. Only known prefix events enter
validation aggregates; target-reader events are excluded entirely. The first
one-gap peer event is excluded where a prior reader pace is unavailable.

Compare raw/log-target MAE LightGBM, user-balanced MAE LightGBM, log-target
CatBoost, and a book-date-anchored MAE correction. Completed standalone regressors are reported even when weaker. The optional
categorical/balanced trials hit an interrupted save; their retries were stopped
to prioritize production exports, so no evaluation score is claimed for them.
The categorical retry code caps CTR complexity/table size and iterations.
Incomplete caches were removed; completed accepted models were preserved.

The production calibration candidate uses the **accepted forecast** plus all
history, book-date and peer features to predict a residual in days. Fit depths
3/5 with MAE on penultimate-event examples; use an inner 80/20 user split to
choose iterations, mixture and correction shrinkage. Evaluate on final events.
Refit on both sets of out-of-sample historical forecasts for production, then
add the correction to the accepted full-data forecast and clip at zero. Export
seconds, including a half-strength alternative. The final refit uses only
known training labels. No test outcomes are available or used.

Task 6 input book IDs and known other-reader dates are legitimate features;
there is no reconstruction of hashes, external Goodreads collection or
cross-task target matching. All rating-zero events remain in Tasks 5 and 6.

## Reproduction and checks

These are incremental experiments using previous accepted caches, not a
standalone cache-free assignment entry point. Requires NumPy, pandas, SciPy,
CatBoost, LightGBM and g++. Use a fresh output directory after configuration
changes. Prior stages are described in TASK1_IMPROVE.md and TASK5_IMPROVE.md
and TASK6_IMPROVE.md.

Task 1: run `task1_next.py` for workers `svdpp`, `svdppstrong`, `itemspp` and
`neighbors`, both with and without `--full`. Then `task1_next_stack.py --predict`;
`--original` is the original SVD/cosine forecast and `--accepted` is the newer
accepted SVD++ ensemble forecast. `--previous` points to its model cache.

Task 5: run `task5_next.py`, `task5_ease_next.py` and `task5_fossil_next.py` for
validation and `--full`, then `task5_next_stack.py --predict`. `--baseline` is
the original Task 5 artifact directory; `--previous` is the accepted improved
sequence ensemble. Both include caches generated by the existing scripts.

Task 6: run `task6_next.py` with and without `--full` to build peer features,
then `task6_next_stack.py --predict`. `--previous` is the accepted Task 6
artifact directory; its parent is the legacy cache root. `--accepted` is the
accepted full-data CSV. The optional `task6_next_models.py` records the other
regressor comparisons; it is not needed to reproduce the residual-calibration
forecast. Set PYTHONPATH to the installed LightGBM package directory if needed.

`next_checks.py` checks metric/FOSSIL updates against finite differences, peer
feature target perturbation, constrained EASE optima, saved-model/component
replay and CSV alignment. `audit_exports(repo, cache_root)` verifies the chosen
primary exports. Full datasets/model binaries stay local. Only reproducible,
superseded intermediate Task 3 checkpoints were removed to free local disk;
accepted forecasts and input datasets were retained.

Local results and export hashes are in the accompanying round-two JSON files.
A local gain is not a promise about the public or hidden half of the test set.

## Measured local comparisons

| Task | Accepted local reference | New primary |
| --- | ---: | ---: |
| 1, MSE | 0.5384875114467605 | 0.537658596135082 |
| 5, accuracy | 0.911909755029613 | 0.9166916322909225 |
| 6, MAE seconds | 7,419,901.322320926 | 7,155,338.850502782 |

These are the same fixed local cohorts used in the earlier round. The new full-
data refits have no separate local score; they produce the upload files.
Task 1's selected forecast is 75% new/25% accepted. Task 5's selected forecast
is 25% depth-5/75% depth-7, without reference shrinkage. Task 6 selects the
depth-5, 507-tree residual calibration at full strength.
