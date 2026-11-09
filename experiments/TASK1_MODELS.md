# Task 1 additional model and ensemble experiments

Same positive-rating data and fixed 90/5/5 split as TASK1.md. Zero ratings are excluded. Evaluation contains 91,701 held-out rows; public test answers are unavailable and never used.

## Measured evaluation MSE

- Original downloadable SVD/item-neighborhood ensemble: 0.554263
- Expanded nonnegative weighted ensemble: 0.548973
- NMF (biased, 20 factors, 50 epochs): 2.932745; rejected
- SVD++ (10 factors, 10 epochs, regularization 0.1): 0.604637 alone
- Direct CatBoost (depth 6, 700 iterations): 0.645780 alone; depth-8 trial deferred after preserving the completed depth-6 model and predictions
- Nonlinear stacker without direct CatBoost: depth 2 0.543977; depth 4 0.540020; depth 6 0.539109
- Nonlinear depth-6 stacker with direct CatBoost included: **0.538577**

The final stacker improves the original ensemble by about 2.8% on this benchmark. It has not reached the leaderboard reference of 0.503544. Repeated exploratory use of the evaluation partition limits its role as a pristine final test.

## Methods

SVD++ incorporates the identities of books a user has rated as implicit feedback, as described in the recommendation slides. NMF constrains latent factors to be nonnegative. A poor result for these particular settings does not establish that the whole model family is unsuitable.

Direct CatBoost learns nonlinear patterns from user and item categories, training-history activity counts and aggregated timestamp summaries. Prediction inputs do not include held-out interaction timestamps or rating aggregates. CatBoost's own ordered categorical encodings use training targets only.

The expanded weighted ensemble fits nonnegative weights summing to one on the 5% selection partition. Its candidates include SVD variants, item/user similarity models, different neighborhood sizes, similarity powers and learned ridge neighborhood weights. Models with selection MSE >= 0.64 are excluded from weighted blending.

The nonlinear stacker uses base-model predictions plus user/item training-rating count, mean and standard deviation. Base models see only the initial 90% training partition. To tune the stacker, the 5% selection partition is split 80/20 with seed 17. Depth and iteration choices use this internal selection split, then the stacker is refitted on all of the 5% selection partition. The final 5% evaluation ratings are never fitted. The final with-direct-CatBoost check uses the already selected depth 6; its internal-selection MSE improves from 0.530118 to 0.529247.

Candidates with selection MSE >= 1 are excluded from stacking, removing the poor NMF setting. Direct CatBoost is included in stacking despite being weaker alone.

## Reproduce

Install numpy, pandas, scipy, scikit-surprise and catboost. First run the Task 1 scripts documented in TASK1.md, then:

```bash
python experiments/task1_newmodels.py
python experiments/task1_catboost.py
python experiments/task1_expanded_blend.py
python experiments/task1_stacking.py
```

The scripts use ignored experiments/artifacts for predictions, fitted CatBoost models and results. The stacking script evaluates depths 2, 4 and 6 with all available component predictions; the recorded final with-CatBoost experiment evaluated depth 6 only. SVD++ and direct categorical CatBoost can take several minutes.

These are experiment runners. The notebook and task1_submission.py still generate the original 0.554263 ensemble; neither they nor the earlier downloaded prediction CSV have been silently replaced with the experimental stacker. A production stacker needs its full training/inference pipeline prepared before a new submission CSV can be claimed.
