# Tasks 2, 4, 5: complementary pair rankers

The accepted public accuracies entering this round are 0.9505, 0.9466, and 0.9219689811720785. User-supplied top-ten targets are 0.9545, 0.9517, and 0.9295278040236905. Task 1 is unchanged.

Only supplied data is used. All observed interactions, including zero ratings, remain in these three tasks. Task 2 uses reading presence. Task 4 uses known timestamps. Task 5 uses sequence order, never numeric timestamps or ratings.

## Models

Retain the accepted component score banks and compare leaf-wise LightGBM classifiers (15/31/63 leaves, regularized leaves, logistic loss) with two neural pair classifiers (64/32 and 128/64 hidden layers). Signed asinh compression reduces the influence of extreme count features. Mirrored candidate pairs guarantee that labels reflect candidate order, and final scores are antisymmetric: P(first beats second) minus P(second beats first).

All three tasks additionally test full-history user neighborhoods. Each reader is represented by their binary history with book weight 1/sqrt(popularity+10); normalize each vector to unit length. Exclude the query reader from their own neighbors. For K=50,200,800, score candidate books by neighbor similarity sum, maximum and voter count. Task 4 also weights these votes by exp(-abs(neighbor interaction time-query time)/h), h=30/180/730 days. All similarity graphs and date votes use only the fitting partition. Task 5 uses each fitted user prefix for retrieval in both validation cohorts, excluding both held-out events. It complements the existing recent-session neighborhood model. Unknown candidate books receive zero neighborhood votes.

## Validation

Reuse the previous reproducible fitting/selection/evaluation splits and negatives. Split the selection partition 80/20 before mirroring, seed17. All architecture, stopping and convex blend selection use this inner split. Refit selected combiners using the selection partition only, then measure evaluation accuracy. Task 5 keeps per-user penultimate/final-event validation. Tasks 2 and 4 use the established random-event holdout. Test labels are unavailable and are never used. These local evaluation cohorts have been reused across experiments; their scores are not unbiased estimates of final hidden-test performance.

## Reproduction

Run `tasks245_round4.py --task 2` (and 4,5). Reader features: `tasks24_reader_neighbors.py --task 2` and `--task 2 --full` (likewise 4 and 5); then `tasks245_round4.py --task 2 --readers`. Existing component caches are built by the earlier experiment scripts. The script uses /workspace/work paths for this saved experiment environment.

Model checkpoints and CSVs stay local. Only code, metadata and results are published to codex. Reproducible rejected Task 3 cache files were removed to make room, recorded in round4_cache_cleanup.json; source, original datasets, accepted models and predictions were preserved.

## Audit evidence

Independent dense toy calculations reproduce similarity and temporal votes. An independent all-reader matrix-vector calculation replayed 32 Task 2 queries. All Task 2/4/5 held-out positives are absent from the fitted graph; Task 4 and Task 5 query IDs match the established baseline caches exactly. Output audits replay sampled trained models, verify each accepted reference CSV, and check every exported candidate ID and row order.
