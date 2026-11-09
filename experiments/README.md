> Latest Task 6 baseline: see [TASK6.md](TASK6.md) and run `task6_pipeline.py --predict`. The material below describes earlier history-only experiments; their recorded results are retained in `task6_previous_experiment_results.json`.

# Task 6 experiments

Run from the repository root (Python 3.11+):

```bash
pip install numpy catboost
python experiments/task6.py
```

The first run creates a feature cache under `experiments/artifacts/`. This directory is ignored by Git. The supplied CSV files are never modified.

## Protocol

For each user, reserve the penultimate gap for model selection and the final observed gap for evaluation. Training examples use only earlier targets, sampling up to five per user. Features are computed solely from the history preceding each target. There are 192,034 training examples and 55,403 users in each evaluation set. Targets are trained in days and metrics reported in seconds (MAE) and seconds squared (MSE).

Features: five most recent gaps; count; historical median, mean, standard deviation, minimum, maximum and quartiles; recent-five median, mean and standard deviation; last timestamp; position within a week; elapsed observed history.

Candidates: CatBoost MAE with seeds 42 and 17, log1p-target MAE, RMSE, and an ensemble. Ensemble weights are selected on the penultimate-gap set, never using final-gap targets. Seed averaging and median are also reported.

The final-gap evaluation has been inspected in earlier exploratory rounds. It is a local benchmark, not a pristine final test or a prediction of hidden leaderboard rank. Actual next gaps after the supplied history may follow a different distribution.

## Course references

`Slides/temporal.pdf`: sliding-window/autoregression features and recency weighting.
`Slides/regression.pdf`: regression, temporal features and regularization.
Gradient boosting is an additional method; the implementation does not claim it is covered by those slides.

## Rating-task clarification

The instructor clarification provided by the user states Assignment 1 Tasks 1 and 3 skip rating-zero targets, while Homework 3 includes them. Exclude zero ratings when fitting/evaluating Assignment 1 rating models. Keep interactions for Task 6 gap histories regardless of their rating.

## Outputs

The runner writes fitted candidate models, selection/evaluation metrics and cached predictions to the artifact directory. `task6_results.json` in this directory records the committed experiment run. Model and feature artifacts are not committed. This runner does not overwrite the existing notebook or produce a submission CSV.
