# Task 5: sequential next-book baseline ensemble

The user's accepted Task 4 Gradescope accuracy is 0.9403. Tasks 1-4 entry points remain unchanged.

## Methods from the slides

Slides/temporal.pdf distinguishes temporal from sequential recommendation, introduces Markov chains and FPMC, and discusses combining persistent user preferences with the previous item. This implementation uses ordering, not absolute timestamps:

- Long-term BPR-32 (100 iterations, lr .01) and BPR-64 (150 iterations, lr .03), reg .01.
- Confidence-weighted ALS-64 (20 iterations, alpha20, reg .1). User vectors are folded in from the actual available query history, and separately from its last five books, using frozen trained item factors.
- Directed adjacent-book transition counts, reverse counts and popularity-adjusted transition lift.
- Implicit ALS-32 on the item-to-next-item transition matrix (20 iterations, alpha10, reg .1), scored from the latest book and a decayed last-three-book context.
- Item-cosine similarities from the full history and last 1/3/5 books, with position decay .7.
- Counts/fractions of books among each training user's last 1/5 observations (ordering-based tail popularity).
- Global popularity, history length, already-interacted and cold-item indicators.

The personal + factorized-Markov combination is FPMC-inspired. The components are fitted separately and blended; this is not a jointly trained full FPMC implementation.

No numerical timestamps or ratings are loaded by the model: the CSV reader requests only user/item columns. File order supplies each user's sequence. All interactions, including rating-zero rows, are retained. No book contents or ID-hash interpretation is used.

## Chronological validation

Data: 2,554,047 observations, 86,786 users, 232,427 observed book IDs; histories range from 4 to 2,843 items. User/book pairs are unique and each user's rows form one ordered block.

Reserve the last two observed interactions per user. Global factor models, similarities and transition graphs fit only the earlier 2,380,475 interactions. There is one selection pair per user, with the penultimate event as positive and the earlier prefix as context. There is one evaluation pair per user, with the final observed event as positive and the prefix plus penultimate event as context. Thus each set has 86,786 pairs.

Evaluation ALS vectors are recalculated from that advanced context, while global item parameters remain frozen. BPR user factors remain fitted to the earlier prefix. Global graphs exclude both held-out events, even though the penultimate event is legitimately available in the final event's local context. This can conservatively understate what full-history refitting can learn; no final target is fitted or included in its context.

Negatives are drawn from earlier training interactions, making item sampling popularity-weighted. Previously seen negatives are allowed; only the exact positive item is rejected. Seed43 controls negative sampling. Selection pairs tune the combiner; final pairs are not fitted.

The supplied test contains one pair per user and 3,441 pairs with one globally unseen book (none with two). The evaluation cold-positive rate is 4.06%, close to the test's 3.96% of pairs containing an unseen candidate. Test candidate median popularities are 101/100 versus 91 for all training events and 126 for each user's final observed event. These unlabeled checks support, but do not establish, the hidden sampler.

Candidates absent from the fitted prefix receive neutral latent/similarity scores plus a learned cold-item indicator. This is applied identically to held-out cold books and unseen test IDs. There is no random item fallback and no hardcoded claim that unseen candidates are always positive. The assumed negative sampler yields known negatives, so cold-item calibration must still be checked on Gradescope.

## Combination and held-out results

A mirrored, intercept-free regularized logistic combiner is compared with the shallow pair-classification implementation shared with Task 4. Only code is shared, not fitted models. The classifier sees pair differences and sums and scores both candidate orientations, enforcing antisymmetry. Internal 80/20 selection uses seed17. Depth3/5 candidates are compared using internal selection accuracy only.

| Model | Final-event accuracy |
| --- | ---: |
| BPR-32 | 83.7843% |
| BPR-64 | 85.5795% |
| Static preference ensemble | 87.9001% |
| Sequential linear ensemble | 90.2576% |
| Selected sequential stacker | **90.8511%** |

Ablations of the linear blend:
- Without Markov components: 90.0952%.
- Without recency components: 89.6907%.

The selected stacker has depth5 and 364 iterations, learning_rate .04, l2_leaf_reg10, seed42. Internal selection accuracy is 91.1683%, versus the linear blend's 90.7766%. The final-event evaluation score is reported independently.

Local results use the earlier-prefix training design and are not a leaderboard score. The full refit uses all supplied observations. A public submission is needed before claiming an improvement over the previous Task 5 leaderboard baseline or a particular rank.

## Run

From repository root:

```bash
pip install implicit numpy pandas scipy scikit-learn catboost
python experiments/task5_ensemble.py \
  --data-dir 'Assignment 1/assignment1_2026' \
  --output-dir experiments/artifacts/task5 \
  --predict --stack
```

Omit --stack for the linear combiner, or --predict for validation only. The notebook's Task 5 cell invokes the selected pipeline. The shared task4_stacking.py file is required and is present on codex.

The full pipeline writes 86,786 chosen item IDs to task_5_predictions.csv. Header, input IDs/order and candidate membership were checked, as was pair-swap antisymmetry. Input CSV files are unchanged. Fitted factors and score arrays are cached under the ignored artifact directory for reproducibility and follow-up diagnostics. See task5_results.json for all metrics, coefficients and checks.
