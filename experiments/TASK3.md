# Task 3 temporal rating-prediction baseline

Task 2 is accepted at the user's reported Gradescope accuracy of 0.9461. Its code and Task 1's selected SVD/neighborhood implementation are retained.

## Lecture methods implemented

Slides/temporal.pdf describes item-time bins, per-user signed time deviation with exponent 0.4, gradual reviewer drift and per-day/session user biases. Slides/recommendation.pdf describes extending rating models with time-dependent parameters, similarity-based rating prediction and regularization.

Implemented components:
- Joint ridge-regularized user/item biases, global 30-day bins, item 90/180-day biases, user 90/180-day biases, user-day offsets and user drift.
- Drift uses sign(t - mean_user_training_time) * abs((t - mean_user_training_time)/year)^0.4, capped at +/-5. Separate positive/negative drift variants test asymmetric change.
- Static SVD (16/32 factors, 50 epochs, reg_all=.1) and the notebook's post-fit global calendar-month correction.
- Item-cosine residual neighborhoods on top of static or temporal biases, including optional absolute-date exponential decay (90/365 days).
- Shrunk user-time residual smoothing kernels (7/30/90/365 days).
- A nonnegative weighted ensemble and a shallow nonlinear CatBoost stacker.

This is not a full TimeSVD++ implementation: latent factors themselves do not drift. It implements the lecture's temporal-bias models and combines them with static collaborative models.

## Data and validation

Exclude all 23,888 rating-zero rows, as the Assignment 1 instructor clarification requires. Keep 1,405,354 positive-rated rows. Fixed seed-42 permutation gives 1,264,818 base-training / 70,268 selection / 70,268 evaluation rows (90/5/5).

The supplied test dates fall inside the corresponding user's training-date range in 94.33% of cases, before it in 2.82%, and after it in 2.75%. This supports an interpolation-style random rating holdout; Task 3 is not treated as a next-interaction forecasting task. All known training interactions, including ones after a query date, may inform interpolation. No hidden test ratings are available or fitted.

User drift centers are computed on base-training dates only. Dates outside [1970,2100] are excluded from those centers, use zero drift and invalid-time bin keys, while their ratings remain in static fitting. This prevents extreme erroneous dates from dominating drift centers. Time bins are fixed epoch-aligned day intervals, not all calendar months; the reproduced original month correction separately uses UTC calendar months.

The weighted ensemble fits coefficients only on selection ratings. The stacker uses the 48 base predictions plus 23 numeric query-date and training-history aggregate features (user/item rating and time summaries). Base models never see selection/evaluation ratings. A seed-17 80/20 split inside selection chooses stacker depth/iterations and compares it with an equivalently fitted weighted blend; the selected stacker is then refitted on all selection rows. Final 5% evaluation ratings remain unfitted.

## Local results (MSE)

| Method | Evaluation MSE |
| --- | ---: |
| Static user/item biases | 0.750420 |
| Filtered SVD-16 | 0.749904 |
| Filtered SVD + global calendar-month correction | 0.748198 |
| Joint regularized temporal biases | 0.731756 |
| Temporal bias + item neighborhood, 20 neighbors | 0.717639 |
| Weighted component ensemble | 0.704638 |
| Selected shallow temporal stacker | **0.695851** |

Stacker depths 2/4/6 yielded evaluation MSE 0.701052/0.697675/0.695851. Depth 6 was selected by the internal selection criterion (0.699877 versus weighted blend 0.706670 on the same internal held-out subset), not by evaluation MSE. It uses 698 trees, learning_rate .03, l2_leaf_reg 10, random_seed 42.

All 48 reusable component predictions exactly reproduced the independent experimental prediction arrays (maximum difference 0.0). The full-data refit generated 20,000 test predictions; header, input columns, timestamps, order, finite values and rating range [1,5] were checked.

These are exploratory local results. The reported existing Gradescope baseline is 0.7872095791695498 and the leaderboard reference is 0.6588089447638401; they evaluate different examples. This experiment does not establish a leaderboard score or top-20 rank.

## Reproduce and predict

From repository root:

```bash
pip install numpy pandas scipy scikit-surprise catboost
python experiments/task3_pipeline.py \
  --data-dir 'Assignment 1/assignment1_2026' \
  --output-dir experiments/artifacts/task3 \
  --predict
```

The runner performs the reproducible component experiments, selects the weighted or shallow stacking model, then refits base components on all positive training ratings and generates task_3_predictions.csv. Intermediate artifacts are cached under a data/source/package fingerprint. The current Task 3 notebook cell invokes this pipeline and copies the resulting CSV beside the input CSVs. It leaves Tasks 1 and 2 entry points unchanged.

Source CSVs are not modified. Predictions/models/cache files remain ignored by Git. task3_results.json records component searches, model selection and parity checks. task3_configs.json records the temporal-bias configurations. Reproducing all experiments can take several minutes.
