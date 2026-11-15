# Task 5 round 3: order graphs, neural ranking and preference adaptation

Accepted public reference: **0.9136957573802226 accuracy**, rank 249. Its local
final-event accuracy is 0.9166916322909225. New files are separate candidates;
Tasks 1--4 and 6 prediction files are unchanged.

## Models and features

Retain the entire accepted feature bank. Add:

- **Skip-transition graphs:** known book-to-book transitions within 1/3/10
  positions, weighted by inverse distance and inverse square root of reader
  history length. Score forward/reverse probabilities, recent context, lift and
  directional evidence. Exact two-hop diffusion on the three-position graph
  smooths paths through intermediate books without materializing its square.
- **Community reading order:** fit scalar book scores using a regularized graph
  Laplacian. For directed transition matrix H, solve
  `(diag(in+out+penalty) - H - H.T) theta = in-out`, penalties .1/1/10.
  Candidate scores measure position relative to known history and extrapolated
  trends over the last 5/20 books. This is inferred order, not an observed date.
- **Candidate-aware attention:** each possible next book attends separately to
  the latest ten known books. With `v_h = E[h] dot V[candidate]`, attention is
  `softmax(v_h/sqrt(k) + position_bias)`. Its score is the weighted mean of v_h,
  plus a bias and frozen long-term preference score. Warm-start 32-dimensional
  factors from prefix-fitted FPMC; train 3/10/20 epochs with BPR, reader-balanced
  sampling and a mixture of full-prefix/recent training targets. Export the
  unrefined checkpoint too. Cold, untrained source embeddings are neutral.
- **Larger BPR preference models:** 128 factors, regularization .03, learning
  rate .03, checkpoints 50/150/350; and 96 factors, regularization .003,
  learning rate .05, checkpoints 50/150/300. Use implicit CPU BPR, fixed seeds,
  one thread. Also score latent cosines from latest book/recent-three/recent-20
  contexts. These are fitted on this task's training prefixes only.
- **Query-user adaptation:** freeze the 64-factor FPMC preference item vectors
  and refine a reader vector from their known latest 3/10/all books, with
  an anchoring penalty toward its original vector. BPR steps 25/100; no supplied
  candidate is treated as a positive label. Unknown history books are skipped.

All observed events, including rating-zero rows, remain. Model readers load
only user/item columns. No numeric timestamp, external metadata, hash decoding,
cross-task matching or test outcome is used.

## Combiners and selection

Reuse the accepted chronological design: globally reserve the final two events
per reader. Train global components on the earlier prefixes. Penultimate events
supply 86,786 selection pairs; final events supply 86,786 evaluation pairs.
Use the same seed-43 popularity-weighted negative candidates and seed-17 inner
reader split as before. Split readers before mirroring candidate order.

Compare CatBoost depths 5/7 and an MLP ranker over pair differences/sums. The
selected first-stage model is a 64/32 ReLU MLP, Adam, alpha .1, 16 epochs,
seed 71. It improves the local final-event accuracy to 0.9206323600580739.

Add the new BPR/adaptation features to that bank. Compare MLPs 64/32 (alpha .1,
seed 73) and 128/64 (alpha 1, seed 117). The smaller, eight-epoch model wins.
Inner-only mixture selection chooses **50% new preference ranker / 50% first-
stage sequence ranker**. All ranking margins are antisymmetric: swapping the
candidates reverses the margin.

| Predictor | Local final-event accuracy |
| --- | ---: |
| Previously accepted ensemble | 0.9166916322909225 |
| Order/attention neural combiner | 0.9206323600580739 |
| Preference-augmented neural ranker alone | 0.9231903763279792 |
| Final selected blend | **0.9235245316064803** |

The new components differ in standalone strength; this is a combined-feature
and combiner improvement, not evidence that attention or inferred order alone
outperforms the accepted model. These fixed local cohorts have been used in
previous rounds; public/hidden scores remain unknown for the new files.

## Upload variants

- `primary`: the selected preference/sequence blend.
- `conservative`: half primary, half the accepted public-reference margin.
- `sequence_neural`: first-stage candidate, provided while preference fits ran.
- `larger_refit`: experimental production meta-refit on both sets of historical
  out-of-sample examples. Halve epoch counts to roughly preserve the SGD update
  budget. Its final-event labels now join training, so **no independent local
  accuracy is claimed** for this variant. All these labels come from provided
  training data. It is a separate leaderboard experiment.

The unbudgeted first-stage larger-data refit is retained as an exploratory
local artifact, outside the main upload bundle.

## Reproduction

Requires NumPy, pandas, SciPy, scikit-learn, CatBoost, implicit CPU and g++17.
These are incremental experiments using the existing accepted caches; previous
stages are documented in TASK5.md, TASK5_IMPROVE.md and TASKS_156_ROUND2.md.
Use a fresh output directory after source/configuration changes.

Run each feature worker for validation and `--full`:
`task5_order.py`, `task5_era.py`, `task5_attention.py`, `task5_fold.py`,
`task5_bpr_round3.py`. Common flags: `--data-dir DATA --baseline ORIGINAL_CACHE
--out NEW_CACHE`; attention/fold additionally use `--previous IMPROVED_CACHE`.
BPR requires the installed implicit package on PYTHONPATH if not global.

Run `task5_round3.py --predict`, then `task5_preference_round3.py --predict`,
then optionally `task5_preference_refit.py`. Their common arguments are:
`--data-dir DATA --baseline ORIGINAL_CACHE --previous IMPROVED_CACHE
--reference ROUND2_CACHE --out NEW_CACHE --accepted ROUND2_CSV`.

Large global BPR factor tables are regenerated deterministically rather than
retained; query-score caches and sampled factor pairs are saved for replay.
Input datasets and accepted forecasts remain intact. Rejected Task 3 model
caches were removed locally for disk space; no input data was deleted.

## Verification

`task5_round3_checks.py` checks skip counts against sequence enumeration,
two-hop scores against dense P², community-order solutions against dense
linear algebra, and attention/adaptation updates against finite differences.
Its audit functions check chronological target exclusion, parameter replay,
pair-swap antisymmetry, and exact CSV IDs/order and candidate membership.
Every exported file has 86,786 rows. Results, configuration, source hashes and
file hashes are in task5_round3_results.json. Prediction CSVs/model binaries
remain outside Git.
