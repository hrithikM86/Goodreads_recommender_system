# Task 1: five-class neural rating prediction

Accepted public reference: **0.5368518492394629 MSE**, rank 208, from the
user's latest report. This experiment exports new candidates and preserves
that accepted forecast. Task 5 is unchanged in this round.

## Classification and the rating estimate

The five output classes are ratings 1, 2, 3, 4 and 5. A softmax network outputs
`p(r | user, book, features)`. The CSV rating is the expected value
`sum(r * p(r))`, rather than the most probable class. The conditional mean is
the Bayes prediction for squared error, matching the assignment's MSE metric.
Rating-zero rows are excluded from fitting and local evaluation.

## Neural models tested

**Embedding classifier.** Trainable reader/book embeddings use
`h = tanh(P[user] * Q[book])`, elementwise multiplication, then five logits
`b[class] + bu[user,class] + bi[book,class] + W[class] dot h` and softmax.
24-dimensional embeddings start from the existing SVD++-24, epoch-50 fit.
Training uses sparse per-observation AdaGrad, smoothed class-bias initialization,
and CE alone or `CE + 2 * (expected_rating - rating)^2`. Checkpoints are
3/8/15/25 epochs. All five classes are considered in every update.

The direct embedding classifiers were weaker standalone (local MSE roughly
0.74--1.01 across checkpoints). Their class probabilities remain available as
complementary input signals; no standalone performance improvement is claimed.

**Neural probability combiner.** Production uses neural stacking with a
102-dimensional collaborative feature bank: the accepted 44 signals, ten
training-only reader/book class frequencies, and the embedding classifiers'
probabilities/means. Class frequencies use a global-prior pseudocount of 20.
Input counts are log transformed; means/scales fit only the meta-training fold.

Compare three MLPs: hidden layers 64/32 with CE (seed 42), 64/32 with CE+2*MSE
(seed 73), and 128/64 with CE+2*MSE (seed 117). Each has ReLU hidden units,
10% dropout, five softmax outputs, Adam, and weight regularization. Logits add
a fixed Gaussian prior centered on the older SVD/cosine forecast, variance .7.
This supplies a collaborative starting estimate while the network learns
rating-distribution corrections. Expected-rating MSE selects early stopping.
Selected epochs are 19, 18 and 10, respectively.

The neural networks use NumPy/C++ implementations. Dependencies are NumPy,
pandas, SciPy, CatBoost (to replay the accepted reference) and g++; no new large
framework dependency is required.

## Selection and evaluation

Reuse the existing positive-only split: 1,650,615 component-training ratings,
91,701 selection ratings and 91,701 separate evaluation ratings. The seed-17
inner split of selection fits meta-models on 73,360 examples and selects on
18,341. Reference predictions on this inner holdout come from the corresponding
inner-fitted accepted models, ensuring the reference did not fit those labels.
The current reference on evaluation is its previously saved out-of-sample
prediction, not a prediction from a model fitted to evaluation labels.

A nonnegative sum-one least-squares mixture of the three neural means and the
accepted mean is selected exclusively on the inner holdout. Evaluation labels
are used only to measure the resulting forecasts. Production component models
fit all positive ratings; neural meta-models refit both held-out prediction/
label sets after the local comparison. Test labels are never available.

| Predictor | Local evaluation MSE |
| --- | ---: |
| Accepted ensemble | 0.537658596135082 |
| Best single neural classifier | 0.5375834107398987 |
| Single-classifier/reference blend | 0.535727444126859 |
| Three neural classifiers, without final reference blend | 0.5369895100593567 |
| Primary neural/reference ensemble | **0.535648910259627** |

Primary weights: accepted .3906269077; CE-64/32 .1642789009;
joint-64/32 .3534052874; joint-128/64 .09168890395. Thus approximately
61% is the neural forecast and 39% the accepted forecast. Neural-only weights
are renormalized to sum to one. The conservative alternative averages the
primary with the accepted forecast again.

These are fixed local-cohort comparisons. Leaderboard/hidden scores are not
known, and this experiment does not establish that neural models universally
outperform the existing collaborative models.

## Reproduction and export checks

These are incremental scripts using the accepted Task 1 caches, described in
TASK1_IMPROVE.md and TASKS_156_ROUND2.md. Use a fresh neural output directory
after changing configuration. The full training CSV is `tasks_1_2_train.csv`.

1. Run `task1_neural_cf.py --data-dir DATA --previous CACHE/task1_improved
   --out CACHE/task1_neural`, then repeat with `--full`.
2. Run `task1_neural.py` with `--data-dir DATA --legacy CACHE
   --previous CACHE/task1_improved --reference CACHE/task1_next
   --original ORIGINAL_SVD_CSV --out CACHE/task1_neural
   --accepted ACCEPTED_ROUND2_CSV --predict`.
3. Run `task1_neural_ensemble.py` with the same arguments and `--predict`.

`--original` is the older SVD/cosine forecast; `--accepted` is the most recently
accepted round-two primary. Their roles differ. Datasets and fitted binary
models remain local; scripts regenerate the forecasts from known training data.

`task1_neural_checks.py` independently checks the CE/expected-MSE softmax
Jacobian and native AdaGrad updates against finite differences. Its callable
`audit(args)` replays saved embedding and MLP parameters, checks held-out pairs
are absent from CF training, and verifies all 20,000 output IDs/order, valid
rating ranges and normalized class probabilities. Export hashes and source
hashes are recorded in `task1_neural_results.json`.
