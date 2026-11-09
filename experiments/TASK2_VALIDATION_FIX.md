# Task 2 validation correction

The first ensemble scored 0.9056 on Gradescope despite 0.963269 local accuracy. The old validation negative sampler was too easy for the supplied candidate distribution.

## Evidence from inputs, without test answers

Popularity is the number of occurrences of a book in the supplied training file:

| Population | Median popularity | Mean popularity |
| --- | ---: | ---: |
| Held-out positive books | 58 | 208.49 |
| Old uniformly sampled negative books | 18 | 35.83 |
| Supplied test candidate 1 | 58 | 207.43 |
| Supplied test candidate 2 | 57 | 204.56 |

Also, 383 of the 20,000 test pairs have one candidate already present in that user's training history; none have both. The old sampler excluded every known user interaction from negative sampling, omitting this case entirely.

These observations support popularity-weighted, previously-seen-allowed negatives as a better local approximation. They do not prove the exact hidden generator. Model fitting, sampling probabilities, and combiner learning still use supplied training interactions only. No hidden test labels are available.

## Corrected experiment

Retain the same seed-42 90/5/5 split and component hyperparameters from TASK2.md. Sample a random interaction from the 90% training partition and use its item as the negative; this makes sampling probability proportional to training popularity. Reject only the same candidate ID as the positive. Previously observed candidate books are allowed.

Add an already-interacted indicator to candidate features. The linear combiner learns its direction and weight using selection labels; no test labels or hardcoded test answers are used. For a unique user/book training table, held-out positives have not been observed in the remaining training partition, whereas some sampled background items have. This is an evaluation-design assumption to revisit if the task's generator differs.

Selection of C and combiner fitting remain separate from the final 5% evaluation pairs. Model hyperparameters were not changed during this correction.

| Method | Corrected held-out accuracy |
| --- | ---: |
| Original ensemble weights on corrected pairs | 90.8737% |
| BPR-32 | 90.2249% |
| BPR-64 | 91.2508% |
| Recalibrated ensemble without already-interacted feature | 93.2684% |
| Recalibrated ensemble with that feature | **94.3405%** |

The original ensemble's corrected local score is much closer to its 90.56% reported public score. This supports a validation-distribution mismatch rather than treating the old 96.33% result as representative. The replacement still requires a Gradescope submission.

The popularity coefficient changed from positive to negative after recalibration, consistent with correcting the models' popularity preference when both candidates have similar popularity. Coefficients are raw score weights, not normalized percentages.

## Run and inspect

```bash
pip install implicit numpy pandas scipy scikit-learn
python experiments/task2_ensemble.py \
  --data-dir 'Assignment 1/assignment1_2026' \
  --output-dir experiments/artifacts/task2 \
  --predict
```

The existing Task 2 notebook cell calls this same updated runner. Source CSVs are unchanged. Output includes the replacement `task_2_predictions.csv`, model-selection/evaluation margins, both-candidate score arrays, and reusable factor caches. Factor cache keys include the sparse interaction matrix, model hyperparameters and implicit package version. Artifacts remain ignored by Git.

See `task2_validation_diagnostics.json` for the original-model counterfactual and prediction replay check. `task2_results.json` contains corrected coefficients and metrics. Task 1 remains unchanged.

## Completed replay and replacement audit

Recomputing original predictions from the new full-data factor caches and original coefficients reproduced all 20,000 original choices exactly (100% agreement). This check found no full-refit/scoring inconsistency. The corrected combiner changes 1,280 of the choices. Both versions have the same 20,000 input rows, IDs, order and header; every replacement choice is one of its supplied candidate IDs. The prior downloadable file is retained separately.

The no-seen-indicator ablation scored 93.2684%, versus 94.3405% with the feature. Thus recalibrating against popularity-matched negatives improves the model even without the history indicator; the latter contributes about 1.07 percentage points in this local experiment.
