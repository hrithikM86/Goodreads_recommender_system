# Task 5: joint FPMC and stronger recency neighborhoods

Accepted public reference: **0.9093402161638974 accuracy**, original Task 5
sequential stacker. Its chronological local evaluation is 0.9085105892655497.
The accepted notebook implementation stays unchanged until leaderboard feedback.

## Joint model from the temporal slides

`Slides/temporal.pdf` introduces FPMC as a pairwise decomposition of the
user/previous-book/next-book tensor. The user/previous-book term cancels when
comparing two possible next books. We jointly train the remaining terms:

`score(u,j | last) = item_bias[j] + P[u] dot Q[j] + A[last] dot B[j]`.

Optimize the BPR loss `-log sigmoid(score(positive)-score(negative))` using the
same error for both user-preference and transition factors. This is a joint
FPMC implementation, unlike the separately fitted BPR + transition-ALS baseline.
The C++ updates are independently checked against a NumPy single-step gradient.

| Configuration | Factors | Learning rate | Per-update regularization | Negative sampler | User weighting |
| --- | ---: | ---: | ---: | --- | --- |
| fpmc32_uniform | 32 | .02 | .02 | Uniform observed book | Each observed transition |
| fpmc64_mix | 64 | .02 | .03 | 50% uniform / 50% interaction-popularity | Each observed transition |
| fpmc64_useruniform | 64 | .03 | .01 | Same 50/50 mixture | Uniform user, then transition |

Checkpoints: 15/40/100 epochs, learning-rate decay .98 per epoch, seeds 42/17/23.
Negatives must be observed somewhere in the fitted training prefix and absent
from the chosen user's known prefix. Held-out target labels are never consulted
when drawing training negatives. User-balanced sampling reflects the one-query-
per-user evaluation, rather than giving very long histories most training steps.

Features include the preference score, last-book transition score, bias, their
joint score, and a .7-decayed transition profile from the latest three books.
Globally unseen targets have neutral factor scores. Previous books with no fitted
outgoing transition have neutral transition vectors rather than random fallback.
The baseline cold-item feature remains available to the classifier.

The 64-factor transition profile reaches about 79.8% selection accuracy (80.1%
for user-balanced sampling), versus about 69.7% for the old transition-ALS signal.
The preference score remains weaker than baseline BPR, which is retained.

## Neighborhood additions

**Weighted asymmetric item neighborhoods.** Downweight each reader's contribution
by `1/(1 + history_length / mean_length)`. Form the resulting weighted shared-
reader counts. Similarity divides by candidate support to power `1-alpha` and
history-book support to power `alpha`, alpha .25/.5/.75. Compare no shrinkage and
shared-support shrinkage 5. Aggregate by maximum, history mean, last book, recent
3/10 books, and a .9-decayed whole-history profile. The best new single signal
has about 78.8% selection accuracy; this is not the ensemble accuracy.

**Session-style reader neighbors.** A query is its latest 20 known books, weighted
by .9 positional decay and IDF. Compare its cosine against other readers' known
prefixes, excluding the query reader. Use up to 50/200 neighbors. Candidate scores
measure weighted reading occurrence, 5-book-scale exponential tail
activity (implemented as `exp(-distance/5)`), and occurrence within five positions
after the query's last book in each neighbor's prefix. No numeric time is used.

These are user-history sessions, not timestamp-defined web sessions. The graph,
reader positions and training factors use only fitted prefixes. An evaluation
query may use its legitimately known penultimate event locally; the final target
is excluded from both local context and fitted global graphs.

## Validation and combination

All interactions, **including rating-zero rows**, are retained. Only user/book
columns and their file order are loaded; timestamps and rating values are not
features. User/book IDs are opaque categories. No external book data is used.

Reuse the accepted chronological design exactly: reserve each reader's final two
events. Fit global models on the earlier 2,380,475 interactions. Selection uses
the penultimate event, evaluation the final event, 86,786 pairs in each. Negative
evaluation candidates are sampled from fitted-prefix interactions with seed 43,
as in the accepted pipeline. Known previous books may be negative candidates;
the exact positive is rejected. Candidate and user arrays are verified against
the original caches before any comparison.

Retain all 22 accepted component scores and add the new scores. Classifier inputs
are pair differences and sums. Mirror each training pair with reversed order and
label; compute the final antisymmetric margin from both orientations. Split
selection by user/pair 80/20 with seed 17 **before** mirroring. Tune CatBoost
depths 5/7, up to 1000 trees, learning rate .04, L2=15, seed 42. Early stopping uses
oriented accuracy, then checkpoints are compared using actual antisymmetric pair
accuracy on inner tuning only. Select depth mixture and reference shrinkage there.

The accepted reference is refitted on inner-training pairs solely for the inner
comparison, avoiding reuse of a model fitted to those tuning labels. The original
saved accepted classifier supplies the separate evaluation and test reference.
Final new classifiers fit selection pairs only; the evaluation targets remain
excluded. The same selection-trained combiner weights are reused for production
query scoring rather than fitting them twice on identical data. Full-data
components refit all supplied interactions before inference.

Prediction variants: `primary` is the new depth mixture; `blend50` averages its
margin with the accepted classifier's margin. These fixed names distinguish the
two upload candidates from any internally selected shrinkage weight.

## Reproduce

Requires numpy, pandas, scipy, scikit-learn, catboost, implicit and g++. If the
accepted Task 5 caches are missing, first run `task5_ensemble.py --predict --stack`.

```
python experiments/task5_improve_checks.py --out experiments/artifacts/task5_improved --numerical-only
python experiments/task5_improve_stack.py --data-dir 'Assignment 1/assignment1_2026' --baseline experiments/artifacts/task5 --out experiments/artifacts/task5_improved --predict
python experiments/task5_improve_checks.py --data-dir 'Assignment 1/assignment1_2026' --baseline experiments/artifacts/task5 --out experiments/artifacts/task5_improved
```

Use a fresh output directory after changing hyperparameters or source. Data/stage
hashes and stacker configuration guards protect caches. Per-worker filesystem
locks allow safe precomputation without duplicate fitting. Large fitted models,
score arrays and prediction CSVs remain outside Git.

## Local results

Selected new ensemble: 25% depth-5 (386 trees), 75% depth-7 (219 trees).
Inner pair accuracy: 0.9158889273, versus accepted reference 0.9116833737.

| Predictor | Final-event evaluation accuracy |
| --- | ---: |
| Accepted reference | 0.9085105893 |
| New sequential ensemble | **0.9119097550** |
| Fixed 50/50 blend with accepted | 0.9109533796 |

The new ensemble improves this local split by about 0.34 percentage points.
It is not yet a leaderboard result or a claim of reaching rank 50. The full-data
component refit has no independent evaluation score; the table describes the
frozen-prefix fits. Upload candidates separately for public feedback.

## Export verification

Both 86,786-row files preserve input columns/order and select one supplied
candidate on every row. Saved classifiers reproduce the CSVs; the accepted
reference CSV is independently reproduced too. Checks cover BPR gradients,
final FPMC factor scores, reader-intersection item similarities, independently
retrieved session neighbors, chronological target exclusion, cached cohort IDs,
input hashes and pair-swap margin antisymmetry. Feature columns use a fixed
named schema across resumed and fresh fits.
