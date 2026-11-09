# Task 4: temporal interaction-prediction baseline

The user reports Task 3 Gradescope MSE 0.704589533460822. Its selected pipeline is retained. Tasks 1 and 2 also remain unchanged.

## Models and lecture connection

Slides/recommendation.pdf covers BPR, confidence-weighted implicit recommendation, cosine/Jaccard similarities and ranking evaluation. Slides/temporal.pdf discusses time-dependent preferences, short/long-term history weighting and temporal latent factors.

This ensemble combines:
- Static BPR-32 (100 iterations, lr .01) and BPR-64 (200 iterations, lr .03), both reg .01.
- Static implicit ALS-64 (alpha20) and ALS-128 (alpha40), 20 iterations, reg .1.
- Temporal ALS-64 with one preference vector per (user,180-day bin) or (user,365-day bin), alpha20, 20 iterations, reg .1. These are piecewise-constant temporal preference approximations, not full TimeSVD++. Unknown user-periods fall back to the complete static ALS-64 score; latent vectors from different coordinate systems are never mixed.
- Item cosine/Jaccard similarity to known user history, plus exponential absolute-date weighting with 30/180/365-day scales.
- Book interaction counts and fractions within +/-30/90/365 days of the query, distance to its nearest observed training interaction, and signed age since its first observed training interaction.
- Global popularity and already-interacted indicators.

Temporal item statistics exclude nonsensical dates outside [1970,2100]. Those rows still count as static interactions. Zero-rated rows are retained, since this is interaction prediction, not rating prediction.

A regularized logistic score-difference combiner is compared with shallow CatBoost classifiers (depth 3/5). Pair differences and pair sums are used by the nonlinear model. Training pairs are mirrored with reversed labels. Its prediction margin is P(first-positive | differences,sums) minus P(first-positive | -differences,sums), enforcing candidate-order antisymmetry. Selection of combiner type and tree depth uses internal selection accuracy only.

## Validation

Random seed-42 90/5/5 split of all 1,429,242 supplied training interactions: 1,286,317 component-training / 71,462 selection / 71,463 evaluation positives. Each held-out positive is paired with a popularity-weighted negative sampled from training interactions (seed43), allowing previously observed candidates and rejecting only the positive ID itself. The combiner's internal 80/20 selection split uses seed17. Final evaluation pairs are never fitted.

This follows the corrected Task 2 protocol. Unlabeled Task 4 input checks support it: candidate median popularities are 318/318 versus 325 for observed interactions. There are 1,014 test pairs with one already-observed candidate. Query dates fall inside user training histories in 94.66% of test rows. Queries precede the book's first observed interaction in 13.50% of local candidates versus 13.33% of test candidates. These checks do not prove the exact hidden test sampler.

All fitted factors, item-time statistics and similarity graphs use training interactions only. Test user/book IDs and timestamps are used only as prediction inputs. Both earlier and later supplied training interactions can inform time interpolation; Task 4 is not treated as next-interaction forecasting.

## Held-out accuracy

| Model | Accuracy |
| --- | ---: |
| BPR-32 | 83.6923% |
| BPR-64 | 83.3228% |
| Static-only ensemble (recalibrated on this task) | 88.4752% |
| Temporal linear ensemble | 92.8872% |
| Selected temporal nonlinear ensemble | **93.2776%** |

Ablations, refitted on the same selection pairs:
- Without temporal ALS: 92.8508%.
- Without book-time statistics: 90.0144%.

Thus book activity near the query date provided the largest temporal gain. Temporal ALS added only a small marginal gain in this configuration.

The selected stacker has depth5 and 698 iterations, learning_rate .04, l2_leaf_reg10, random_seed42. It won on internal selection accuracy (93.3324% vs linear 93.1575%). Its final evaluation accuracy is reported independently. These are local results, not a Gradescope score.

## Run

From repository root:

```bash
pip install implicit numpy pandas scipy scikit-learn catboost
python experiments/task4_ensemble.py \
  --data-dir 'Assignment 1/assignment1_2026' \
  --output-dir experiments/artifacts/task4 \
  --predict --stack
```

Omit --stack to use only the linear combiner. Omit --predict for validation only. Component models are refitted on all supplied training interactions before test prediction. Factor caches and both-candidate score arrays are retained under the ignored artifact directory for later diagnostics.

The notebook Task 4 cell invokes this runner. Outputs contain the chosen item ID, not 0/1. All 20,000 generated rows passed header, input-column, timestamp, row-order and candidate-membership checks. Candidate swapping was checked to negate the nonlinear score exactly. No missing-ID fallback was needed on this test file.

See task4_results.json for all component metrics, coefficients, ablations, cohort checks and accepted Task 3 public score. Source CSVs are unchanged; generated predictions/models remain ignored by Git. A leaderboard submission is needed to establish actual performance.
