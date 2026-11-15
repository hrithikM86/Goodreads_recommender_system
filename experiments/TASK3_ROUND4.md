# Task 3: wider joint neighborhoods with separate regularization

Accepted public reference: round-3 new ensemble, **0.6825445785868727 MSE**.
Its local evaluation MSE is 0.679167621198195. The notebook retains that accepted
model until a new candidate has a better reported leaderboard result.

## Model

For each target book, fit observed readers' temporal-bias residuals using other
books' binary reading indicators X and rating residuals R:

`min_(a,b) ||target_residual - X a - R b||² + lambda_p ||a||² + lambda_r ||b||²`.

The target book is never one of its own predictors. Fit only readers with a known
positive target-book rating. All 23,888 zero-rated rows are removed before every
fit, history and graph. No other task's labels or hidden test labels are used.

Compare up to **64** and **256** candidate neighbors, reusing the earlier union of
normalized reading-overlap and residual-correlation candidates. These are maxima;
overlapping candidate lists produce smaller neighborhoods. Reading penalties are
10/50/200, and rating-to-reading penalty ratios are .25/1/4/16: 24 configurations,
each with static and 365-day date-weighted predictions. Unlike the earlier joint
model, the reading and rating terms can shrink differently.

The solver whitens the rating columns by `1/sqrt(ratio)`, applies ordinary ridge
with lambda_p, and converts coefficients back. Primal/dual eigensolvers are
numerically checked against direct inversion with the diagonal penalty matrix.
The equal-penalty, 64-neighbor control reproduces the round-3 joint50 predictions
**exactly** for all validation queries, including date-weighted predictions.

## Train-only screening and stacking

Reuse the seed-42 positive-only 90/5/5 split. Component fits use 1,264,818 ratings;
selection and evaluation have 70,268 ratings each. Split selection 80/20 with
seed 17. Choose one lambda_p for each width/ratio family using only the **56,214
inner-training labels**. Retain both date variants. The old equal-penalty control
and support counts already exist in the accepted feature bank, so they are not
added again. This leaves seven new component configurations / 14 new columns.

Train direct-rating CatBoost depths 6/8 (learning rate .025, L2=20, seed 42,
maximum 1800 trees, early stopping 120). Select the depth mixture and shrinkage
toward the accepted ensemble using only the **14,054 inner-tuning labels**.
Depth weights are 0/.25/.5/.75/1; new-model weights are 0/.5/.75/1.

Report performance on the separate, reused 70,268-row evaluation split. After
recording that comparison, refit stackers using all **140,536 held-out training
ratings** and fit components using all **1,405,354 positive training ratings**.
The larger final refit has no remaining independent local holdout score.

## Initial component findings

The original 64-book equal-penalty model remains strongest alone on selection:
0.699315 MSE. Best wider component: 0.702133 (256 neighbors, both penalties 200).
These component scores are diagnostics, not evaluation-set selection criteria.
A fixed 50/50 average of those two predictions scores **0.698602**, suggesting
complementary information despite the weaker standalone wider score.

## Reproduce

Requires the previous round-2 and round-3 caches and their dependencies.

```
python experiments/task3_round4_checks.py --numerical-only
python experiments/task3_round4_stack.py --data-dir 'Assignment 1/assignment1_2026' --legacy experiments/artifacts/task3 --prior experiments/artifacts/task3_optimized --round2 experiments/artifacts/task3_round2 --round3 experiments/artifacts/task3_round3 --out experiments/artifacts/task3_round4 --predict
python experiments/task3_round4_checks.py --data-dir 'Assignment 1/assignment1_2026' --legacy experiments/artifacts/task3 --prior experiments/artifacts/task3_optimized --round2 experiments/artifacts/task3_round2 --round3 experiments/artifacts/task3_round3 --out experiments/artifacts/task3_round4
```

`--legacy` must point to the directory containing `split.npz` (resolve `run.json`
cache indirection if needed). Use a fresh output directory after source or
hyperparameter changes. Data hashes, stage flags and selected-config manifests
guard caches. Saved full-data coefficient matrices permit independent prediction
reconstruction. Predictions and fitted model binaries remain outside Git.

## Selected stack and local evaluation

The new depth-6 stacker stops at 991 trees, inner MSE 0.6823338802. The new
depth-8 stacker stops at 932 trees, inner MSE 0.6817127424. Inner selection chooses
only the new depth-8 stacker, blended **50/50 with the accepted ensemble**;
inner MSE 0.6815147760, versus accepted reference 0.6819525240.

| Predictor | Separate local evaluation MSE |
| --- | ---: |
| Accepted round-3 ensemble | **0.6791676212** |
| Wider/separate-penalty stacker alone | 0.6798768138 |
| Inner-selected 50% new / 50% accepted | 0.6792432164 |
| Fixed lighter 25% new / 75% accepted | 0.6791356685 |

**This experiment does not establish a meaningful improvement.** The lighter
blend's MSE reduction is only 0.00003195, with user-clustered standard error
0.00005875 and approximate conditional 95% interval [-0.00008319, 0.00014710].
This is a reused local split and the interval does not cover broader model-search
or train/test distribution uncertainty. The accepted public model is retained.

Exported files are **exploratory leaderboard candidates**, not proven upgrades:
`raw` is the new wider stacker alone; `primary` is the inner-selected 50/50 blend;
`conservative` is 25% new / 75% accepted. No additional weight is chosen from the
separate evaluation results. Each final stacker uses all 140,536 meta-training
rows, so these local scores describe the earlier 70,268-row stacker fits.

## Reported leaderboard feedback

Updated stacker alone: 0.6838784593036632; 50/50 blend: 0.6829861412822404;
25% new blend: 0.6827090155188005. All are worse than accepted 0.6825445785868727.
The round-3 primary remains accepted.
