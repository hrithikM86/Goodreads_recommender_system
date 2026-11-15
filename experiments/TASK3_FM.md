# Task 3: add time-aware factorization machines to the accepted ensemble

Accepted public reference: **0.6825445785868727 MSE**, round-3 primary.
Round-4 wider-neighborhood models were rejected after all three uploaded variants
scored worse. The notebook remains on the accepted round-3 implementation.

## Actual FM model

These are second-order factorization machines, not renamed SVD predictions:

`prediction = intercept + sum_j w_j*x_j + sum_(j<k) dot(V_j,V_k)*x_j*x_k`.

Sparse fields encode user, book, 90-day calendar period and a user-specific
continuous time-drift feature. Drift is signed absolute deviation from that
user's mean known training date, to power .4 in year units, clipped to [-5,5].
Context variants also include user/180-day and book/180-day categories. The FM
learns every pairwise interaction among the active fields, including book/time,
user/time and book/user-drift effects. Unseen fields are omitted at inference.
Invalid dates do not get artificial calendar categories or drift values.

Profile variants add 12-dimensional normalized reading-profile vectors and
12-dimensional book vectors derived from the earlier **training-only** TF-IDF
reading-graph SVD. These are latent reading patterns, not external book metadata.
The current book is removed from each training row's reader profile; this is
independently unit-tested. Query profiles use only known training interactions.

All 23,888 rating-zero rows are excluded before labels, histories or graphs.
No other task's labels or hidden test ratings are used. Known training dates on
either side of a query may inform date-conditioned rating interpolation.

## SGD models

The C++ kernel implements the exact pairwise equation and simultaneous SGD
gradient; both are checked against independent small NumPy examples. A fixed
training-target mean is the intercept. First-order coefficients have .03
per-active-feature update regularization; factor regularization varies below.
Learning rates decay by .98 per epoch. This is the ordinary sparse SGD
parameterization, not a claim of unweighted global L2 coordinate minimization.

| Configuration | Factors | Target | Factor penalty | Learning rate | Checkpoints |
| --- | ---: | --- | ---: | ---: | --- |
| Basic | 16 | Temporal-bias residual | .05 | .008 | 20/50/80 |
| Context | 16 | Residual | .1 | .005 | 20/50/80 |
| Basic | 32 | Raw rating | .1 | .005 | 20/50/80 |
| Context | 16 | Raw rating | .2 | .005 | 20/50/80 |
| Context | 32 | Residual | .2 | .005 | 20/50/80 |
| Basic, strong shrinkage | 16 | Residual | .2 | .005 | 5/10/20/40 |
| Context, strong shrinkage | 16 | Residual | .4 | .004 | 5/10/20/40 |
| Context, weaker shrinkage | 32 | Raw rating | .03 | .004 | 20/50/80 |
| Reading profile | 16 | Residual | .2 | .005 | 5/10/20/40 |
| Reading profile | 32 | Raw rating | .05 | .005 | 20/50/80 |

The residual baseline is the previously fitted temporal-bias model. It is fitted
on component-training rows only; checkpoint predictions add that baseline back.
Longer weakly regularized fits overfit. The best standalone selection score is
about **0.72435** (basic strong-shrinkage residual FM, 10 epochs). These scores
describe a component, not the complete accepted ensemble.

## Ensemble protocol

Keep the seed-42 positive-only 90/5/5 split: 1,264,818 component-training labels,
70,268 selection labels, 70,268 evaluation labels. Split selection 80/20 with
seed 17. Choose one checkpoint per FM configuration using only the **56,214
inner-training labels**, then add those ten predictions to the accepted 245
features. The **14,054 inner-tuning labels** select CatBoost stopping and mixture
weights. Direct simple blends of each FM (and their mean) with the accepted
inner ensemble are checked too; the SGD direct-blend search chose zero FM weight.

Fit CatBoost depths 6/8, learning rate .025, L2=20, up to 1800 trees, early stopping
120, seed 42. Choose the depth mixture from 0/.25/.5/.75/1 on inner tuning.
Compare the resulting augmented ensemble and its fixed 50/50 average with the
accepted model on the separate evaluation split. Reference shrinkage is also
selected and reported, but exported files are explicitly named `augmented` and
`blend50` so there is no ambiguity about their weights.

After the local comparison, production component fits use all **1,405,354 positive
training labels**, and stackers use all **140,536 held-out training labels**.
There is no independent local holdout left for those larger production refits.
All local comparisons are on the reused validation split, not hidden test data.

## Bayesian FM comparison

`myfm==0.5.0` supplies a separate 16-factor Gibbs-sampled FM on the four basic
temporal fields. Its intercept is learned. User, book, period and drift groups
have separately learned hierarchical regularization. Checkpoints at iterations
30/60/90 average the final five draws of each window. Select the checkpoint using
only inner-training labels. Save its actual intercepts, linear coefficients and
factors so the posterior-window predictions can be independently reconstructed.
This is a short predictive sampling experiment, not a convergence claim.

A final train-only search compares mixtures of accepted, SGD-augmented, and
Bayesian predictions. SGD weights are 0/.25/.5/.75/1 and Bayesian weights are
0/.025/.05/.1/.15/.2/.3, with nonnegative accepted weight and total weight one.
If Bayesian weight is positive, refit that Bayesian checkpoint on full training
data before exporting its blend. If zero, retain the negative result without
pretending Bayesian predictions improved the accepted model.

## Reproduce

Requires prior round-2/round-3 caches and their dependencies, g++, plus optional
`myfm==0.5.0` and tqdm for the Bayesian branch. The SGD branch needs no new ML
library beyond the earlier experiment dependencies.

```
python experiments/task3_fm_checks.py --out experiments/artifacts/task3_fm --numerical-only
python experiments/task3_fm_stack.py --data-dir 'Assignment 1/assignment1_2026' --legacy experiments/artifacts/task3 --prior experiments/artifacts/task3_optimized --round2 experiments/artifacts/task3_round2 --round3 experiments/artifacts/task3_round3 --out experiments/artifacts/task3_fm --predict
python experiments/task3_bayesian_fm.py --data-dir 'Assignment 1/assignment1_2026' --out experiments/artifacts/task3_fm/bayes_validation
python experiments/task3_fm_bayes_blend.py --data-dir 'Assignment 1/assignment1_2026' --legacy experiments/artifacts/task3 --prior experiments/artifacts/task3_optimized --round2 experiments/artifacts/task3_round2 --round3 experiments/artifacts/task3_round3 --out experiments/artifacts/task3_fm --predict
```

`--legacy` must resolve to the directory with `split.npz`. Use fresh output
directories after changing code or hyperparameters. Data, stages, component
selection and stacker configurations are guarded by manifests. Checkpoints and
CSVs stay outside Git; results, source hashes and audits are recorded separately.

## Measured results

The augmented depth-6 stacker stops at 1378 trees (inner MSE 0.6821068254),
and depth 8 at 1072 trees (inner MSE 0.6824693653). Their 50/50 average scores
0.6818916220; a further 50/50 average with the accepted ensemble scores
**0.6817439401** on inner tuning, versus reference 0.6819525240.

| Predictor | Separate local evaluation MSE |
| --- | ---: |
| Accepted round-3 ensemble | 0.6791676212 |
| FM-augmented ensemble | 0.6790911650 |
| Fixed 50/50 augmented + accepted | **0.6789754515** |

The blend reduces local MSE by 0.00019217, with user-clustered standard error
0.00008570 (conditional approximate 95% interval [0.00002421, 0.00036013]).
This is a modest gain on a reused split, not evidence of a particular leaderboard
rank. Full-data production refits have no separate holdout score.

The Bayesian branch selects 60 iterations on inner-training loss. Its separate
evaluation MSE is **0.7351057694**. The joint blend search assigns it **zero
weight**, retaining 50% SGD-FM-augmented and 50% accepted predictions. No full-data
Bayesian model is needed or exported. Its selected validation checkpoint is
independently replayed from five stored parameter draws.

Files: `augmented/task_3_predictions.csv` is the new FM-feature stacker ensemble;
`blend50/task_3_predictions.csv` is its fixed 50/50 average with the accepted
round-3 model. Both production variants use larger final training fits described
above, not the selection-only fits used for the local comparison. The accepted
public model remains unchanged pending leaderboard feedback.

## Reported public feedback

FM-augmented: 0.6833593918541966; 50/50 blend: 0.6828225885165874.
Both are worse than accepted 0.6825445785868727. Per user preference, retain the
model that performs best on known public scores; the round-3 primary remains
accepted. Regenerable rejected SGD-FM parameter checkpoints were removed locally
to make room for Task 5 experiments; input datasets, scores, audits, code and
prediction files were retained.
