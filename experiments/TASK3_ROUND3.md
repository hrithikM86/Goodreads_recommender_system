# Task 3: joint rating/exposure neighborhoods and residual stacking

Accepted reference: round-2 fixed blend, **0.6855387514011825 public MSE**,
reported Task 3 rank 123. Its primary scored 0.6862066239120452.
Rank-50 target supplied by the user: 0.6750716. Local scores do not establish a
leaderboard rank, and no test labels are available.

## New components

All components exclude rating-zero rows before fitting, histories, similarities
and aggregates. They use only `tasks_3_4_train.csv`; test user/book/date columns
are inference inputs. Other tasks are unchanged.

**User-specific ridge neighborhoods.** Transpose the previous observed-rating
regression: for each reader, select up to 64 other readers by normalized reading
overlap and residual correlation. On books this reader actually rated, fit a
signed ridge regression to their temporal-bias residuals. Predict using other
readers' ratings of the requested book. The reader cannot be their own neighbor.
Rating residuals and binary exposure are tried separately, with penalties
5/25/100 for ratings and 25/100 for exposure. Static and 365-day weighted outputs
are features. Standalone selection MSE: best binary-exposure variant **0.717997**.

**Joint book neighborhoods.** The previous model fit binary exposure and rating
residual predictors separately. The new model fits both together:

`w_j = argmin_w ||(r_j - temporal_bias)_observed - [exposure, residuals] w||² + lambda ||w||²`.

This lets reading a neighboring book have one effect and the rating assigned to
it have another, with both coefficients adjusted jointly. Only readers of the
target book supply labels. The target book is excluded from predictors. Candidate
neighbors are the same as the round-2 64-book model; penalties are 10 and 50.
Static and date-weighted outputs are included. Best standalone selection MSE:
**0.699315**, versus about 0.705759 for the earlier rating-only neighborhood.

**Categorical book neighborhoods.** Replace each neighbor's linear rating term
with five indicators, one for each possible rating. Fit the same observed-only
ridge objective. This allows nonlinear relationships between neighboring ratings.
The best standalone selection variant scored **0.705059**. It is supplied as a
potential complementary signal rather than claimed as the strongest model.

**Residual stacking.** Compare direct rating prediction against predicting
`rating - anchor`, where the anchor is the fixed average of the two existing
365-day rating25 book-neighborhood predictors. The anchor uses no query labels.
CatBoost direct-rating depths 6/8 and residual depth 6, learning rate .025,
L2=20 and up to 2200 trees are selected
with early stopping on the same inner split as earlier rounds. Singles and pairs of these three learners are compared using 25/50/75%
mixture weights. A .0002 MSE penalty for a second learner discourages deploying
an extra model for a negligible inner gain.

| Learner | Inner MSE | Selected trees |
| --- | ---: | ---: |
| Residual, depth 6 | 0.684851 | 1917 |
| Direct, depth 6 | 0.682554 | 1550 |
| Direct, depth 8 | 0.682274 | 908 |
| Selected 50/50 direct-depth blend | **0.681953** | Both |

The residual model is excluded from deployment. The new primary combines the
two direct-rating learners.

A LightGBM residual stack (15 leaves, L2=30, minimum leaf 150) scored 0.684518 on
inner selection, behind the accepted bank's 0.682999, so its wider grid was
stopped and it is not deployed. A residual CatBoost control using only the round-2 features scored 0.684791;
with the new components it scored 0.684851. The deeper residual searches were
stopped before completion because these completed comparisons did not improve
on the accepted 0.682999 inner score.

## Split, selection and final refit

The seed-42 positive-only 90/5/5 split is unchanged. Components fit 1,264,818
ratings. The 70,268 selection rows are split 80/20 with seed 17 to select model
depth, stopping iteration and mixture weight. The next 70,268 evaluation rows
are not used in those fits or inner selection. The accepted round-2 blend is
replayed on the same evaluation rows for comparison.

After the comparison is recorded, the final stackers are refitted using **all
140,536 held-out training ratings**, rather than only the 70,268 selection rows.
Full component features are independently fitted using all 1,405,354 positive
training ratings. This larger stacker refit has no separate reported local score;
the local comparison describes models fitted on selection only. Neither stage
uses hidden test labels. The accepted submission is preserved.

An additional `refit/task_3_predictions.csv` control retrains the two accepted
round-2 bank architectures on all 140,536 meta-training rows. This changes only
the amount of known training data used by the stackers. Its benefit has no
independent local estimate; it is a separate upload candidate, not a measured
improvement.

`primary/task_3_predictions.csv` is the newly selected ensemble after the larger
refit. `conservative/task_3_predictions.csv` is its fixed 50/50 average with the
accepted round-2 blend. The conservative weight is not tuned on evaluation or
leaderboard scores.

## Reproduce

Requires the existing round-2 component caches, Python packages used by round 2,
and a C++ compiler to regenerate those earlier caches. With those caches present:

```
python experiments/task3_round3_stack.py --data-dir 'Assignment 1/assignment1_2026' --legacy experiments/artifacts/task3 --prior experiments/artifacts/task3_optimized --round2 experiments/artifacts/task3_round2 --out experiments/artifacts/task3_round3 --predict
```

Component matrices and fitted ensembles remain outside Git. Data/stage hashes
guard the new component caches. Use a fresh output directory when changing code
or hyperparameters. Component feature files are written atomically. Metrics and
audits are recorded in `task3_round3_results.json`.

## Separate local evaluation

| Model | Evaluation MSE |
| --- | ---: |
| Accepted round-2 blend | 0.6796011133 |
| New direct-depth-6 learner | 0.6793759037 |
| New direct-depth-8 learner | 0.6797434374 |
| Selected 50/50 new-depth ensemble | 0.6791676212 |
| Fixed 50/50 new ensemble + accepted blend | **0.6790349058** |
| Accepted architecture with all-meta refit only | Not independently evaluated |

The gain is modest. For the conservative blend, paired squared-error reduction
is 0.00056621 with user-clustered standard error 0.00012717 (25,842 users, 70,268
rows); the approximate conditional 95% interval is [0.00031696, 0.00081546].
This describes uncertainty within the reused local split, not a guarantee about
the public or hidden evaluation. Production stackers then use the larger
all-meta refit described above; there is no fresh holdout score for that refit.

## Reported leaderboard feedback

New primary: **0.6825445785868727**; conservative blend: 0.6832343420425995;
accepted-architecture larger refit: 0.6835831014779641. The new primary is now
the accepted reference for the next experiment.
