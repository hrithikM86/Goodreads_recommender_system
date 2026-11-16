# Task4: accepted components + Item2vec + FPMC, fresh LightGBM

Public reference: Item2vec-LightGBM accuracy0.9486, rank57. FISM-LightGBM scored0.9477 and is rejected. This pipeline imports the accepted Item2vec bank directly; no FISM features or factors are included.

## FPMC

Train the lecture's Factorizing Personalized Markov Chains model from scratch on Task4 fitting data:

`s(u,i | previous_book) = b_i + P_u dot Q_i + A_previous dot B_i`.

The reader/previous-book term in the full factorized tensor is constant between candidates and can be omitted. Reader preference factors P/Q and transition factors A/B are optimized jointly with pairwise logistic/BPR loss. The native kernel is shared code from our Task5 implementation, but no Task5 data or trained factors are used.

Order each reader's known fitting events by timestamp. Train only consecutive transitions whose timestamps increase strictly. Equal timestamps do not establish a directional transition. For a query, select the latest strictly earlier known fitting event and the earliest strictly later known fitting event. Context is identical for both candidates. Missing/invalid-time context yields zero transition scores, retaining the personal term.

A query's following book is a supplied TRAINING interaction at a later date, not a hidden target or a predicted future event. Task4 is timestamp-conditioned interpolation, unlike Task5's last-event forecasting. We add an additional backward compatibility feature `A_candidate dot B_following` to exploit this legitimate known context. We do not use withheld events to construct either side of a validation query.

Settings are fixed before evaluation:32 factors, seed1049, normal(.0,.05) initialization, learning rate.02, decay.98 per epoch, L2 coefficient.02. Negatives are half active-book uniform and half fitting-event popularity, rejecting all books the reader has already read; skip after100 unsuccessful attempts. Interaction-weighted sampling of transitions is used. Untrained personal/source/target factor rows are zeroed for scoring. Numeric ratings are ignored; zero-rated rows remain observed reading events.

Checkpoints15/40/100 each provide five candidate scores:

1. Personal score including target-book bias.
2. Previous-book transition.
3. Following-book transition compatibility.
4. Canonical personal + previous-book score.
5. Personal + previous/following transition scores weighted by exp(-gap/180 days).

These15 signals are appended to the accepted142 component scores, including Item2vec, for157 candidate scores and314 paired differences/sums.

## Fresh combiner and validation

Reuse the established seed42 fitting/selection/evaluation split and seed43 historical negatives. Validation FPMC fits only the90% component-fitting interactions. Test-time FPMC fits all provided training interactions. The existing component bank is unchanged.

Refit one LightGBM combiner on71462 selection pairs, mirrored to142924 classifier rows, with the accepted settings:15 leaves, minimum100 examples per leaf,1100 trees, learning rate.035, L2 leaf penalty20, row/feature sampling.9, seed619 and two CPU threads. No fixed blend percentages and no evaluation-label tuning are used. Local evaluation has71463 pairs; these cohorts have been reused in earlier experiments, so results are diagnostic rather than unbiased hidden-test estimates. Public accuracy is unknown until user submission.

## Checks

- A finite-difference BPR loss reproduces native updates for personal/transition factors and biases.
- Independent direct score calculations verify all five features.
- Synthetic dates verify strict predecessor/successor behavior, including tied and invalid timestamps.
- Every heldout positive is absent from fitting histories; query IDs match the accepted bank and both candidates receive identical temporal contexts.
- Final factor replay checks64 real queries; serialized LightGBM replay and all20000 CSV rows, timestamps, candidate IDs, score finiteness and antisymmetry are checked.

## Reproduction

```
mkdir -p /workspace/work/task4_fpmc
g++ -std=c++17 -O3 -shared -fPIC experiments/task5_fpmc_kernel.cpp -o /workspace/work/task4_fpmc/fpmc.so
PYTHONPATH=/workspace/work/task2_packages python experiments/task4_fpmc.py
PYTHONPATH=/workspace/work/task2_packages python experiments/task4_fpmc.py --full
PYTHONPATH=/workspace/work/task2_packages python experiments/task4_fpmc_combiner.py
PYTHONPATH=/workspace/work/task2_packages python experiments/task4_fpmc_checks.py --heldout --replay
```

Full final FPMC factors and every checkpoint query-score array are retained. Earlier factors can be regenerated deterministically. Rejected FISM and reader-neighborhood cache files were removed to conserve disk, recorded in task4_fpmc_cleanup.json. Accepted Item2vec-LightGBM artifacts, datasets, source, all prior prediction CSVs and result metadata are retained. Code/results are published only to codex; no large factors, datasets or prediction CSVs are pushed.

## Result

Local evaluation accuracy0.939437191273806 versus accepted Item2vec-LightGBM0.9392132991897906. The new CSV changes130 of20000 predictions. All numerical, context, factor/model replay and output checks passed. Public score is unknown; retain the accepted0.9486 file unless this improves it.
