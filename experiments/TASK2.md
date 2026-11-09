> **Superseded validation protocol:** the 96.33% local result below came from uniformly sampled negatives and did not transfer (reported Gradescope accuracy: 90.56%). The current runner uses the corrected protocol in [TASK2_VALIDATION_FIX.md](TASK2_VALIDATION_FIX.md). The historical runner is available at commit `f6e9d0988010eec6ae6155db8dfcd072919bd1eb`. Historical metrics are preserved in `task2_uniform_validation_results.json`; `task2_results.json` now records the corrected protocol.

# Task 2: interaction-prediction baseline ensemble

## Methods and lecture connection

Slides/recommendation.pdf covers Bayesian Personalized Ranking (relative positive/unobserved preferences), confidence-weighted implicit recommendation (Hu et al.), cosine/Jaccard similarities and ranking evaluation. This runner compares:

- BPR: 32 factors / 100 iterations / learning rate .01 / reg .01 / seed 42, and 64 factors / 200 iterations / learning rate .03 / reg .01 / seed 17.
- Implicit ALS: 64 factors / confidence alpha 20, and 128 factors / alpha 40, both 20 iterations / regularization .1 / seed 42.
- Binary item cosine maximum, sum of top-10 cosine similarities to the user's history, and maximum Jaccard similarity.
- Log item interaction count (popularity).

The ensemble fits a regularized logistic combiner to score differences. Each positive/unobserved selection pair is mirrored with its labels reversed, and the intercept is disabled. This enforces candidate-order symmetry. Scaling is fitted on the selection data. C is selected from .01/.1/1 using an internal 80/20 selection split (seed 17), then the combiner is fitted to all selection pairs. Test scores use the same learned raw coefficients. Missing users/items get a deterministic popularity fallback; ties select item1. Each test row receives exactly one chosen item ID.

All observed rows, including rating-zero rows, count as interactions. No rating values are used. The instructor's zero-rating exclusion applies to rating tasks 1/3.

## Evaluation design

Use all 1,861,643 supplied training interactions. Random seed 42 divides rows into 1,675,478 training / 93,082 selection / 93,083 evaluation positives (90/5/5). Each held-out positive is paired with one uniformly sampled item the user has never interacted with anywhere in the supplied training file (negative-sampling seed 43). Full known histories are used only to reject falsely labeled evaluation negatives; fitted component models and similarity matrices use the 90% training interactions only.

Negative examples are proxies for unobserved interactions, not explicit dislike labels. This local sampling design need not match the hidden test generator. Metrics are pairwise accuracy, with exact ties receiving half credit for local individual-score reporting. Selection pairs fit/tune the combiner; final evaluation pairs are not fitted. There are no supplied test labels.

## Local evaluation accuracy

- Popularity: 75.2672%
- BPR-32: 92.0383%
- BPR-64: 93.9731%
- ALS-64: 93.6358%
- ALS-128: 93.0084%
- Maximum cosine: 92.2290%
- Top-10 cosine sum: 93.1771%
- Maximum Jaccard: 92.1173%
- Ensemble: **96.3269%**

C=.1 was selected internally. Numeric raw score coefficients, feature order and full metrics are recorded in task2_results.json. Coefficients have different scales and should not be interpreted as normalized mixture proportions. Negative coefficients can correct correlated component scores.

## Run

From repository root:

```bash
pip install implicit numpy pandas scipy scikit-learn
python experiments/task2_ensemble.py \
  --data-dir 'Assignment 1/assignment1_2026' \
  --output-dir experiments/artifacts/task2 \
  --predict
```

Without --predict the runner performs validation only. With it, all component models are refitted on the full supplied training data and test predictions are written to experiments/artifacts/task2/task_2_predictions.csv. The test file's user/item1/item2 inputs are used only for final prediction, not model fitting or weight selection. All 20,000 supplied test users and both candidate IDs occur in the training data. A fallback remains for portability.

The Task 2 notebook cell runs this pipeline and copies the CSV back beside the source CSVs. The generic BPR helper's class-name typo and missing continue after fallback are also fixed; later tasks retain their original model approach. Task 1 remains the previously selected SVD/neighborhood ensemble.

Predictions, caches and artifacts remain ignored by Git; source data is unchanged. A public leaderboard result is required before comparing against leaderboard ranks.
