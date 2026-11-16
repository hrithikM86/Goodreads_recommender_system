# Task 5 rolling validation correction

The previous selection and final-event evaluation banks both fitted global components on H[:-2]. Final-event queries nevertheless had H[:-1] as their known context. Test components fitted on H, matching the entire supplied test context. The audit found 3.8197% of final-event validation last-context items had zero graph support, versus zero for test contexts. No own-target history leakage was found.

`task5_rolling.py` retains the accepted selection features from H[:-2] and refreshes **every one of the 260 accepted component-score columns** on H[:-1], with H[-1] excluded from each reader's fitting history. It runs the original base BPR/ALS/Markov/neighborhood models, FPMC variants, stronger neighborhoods, PRME/TransRec, EASE, FOSSIL, positional order/two-hop graphs, inferred reading order, candidate attention, stronger BPR, and query-user fold adaptation. Neither numeric timestamps nor rating values are consumed. Zero-rated interactions remain valid reading events.

Final-event evaluation uses three independent event-frequency negative draws (seeds43,441,997), rejecting the true target only, as in the previous protocol. The accepted sequence and preference networks remain frozen during this evaluation: 224 and260 scores respectively, mirrored differences/sums, 64/32 hidden layers, alpha0.1, seeds71/73, 16/8 epochs, fixed equal average of their antisymmetric probability margins. Scores across negative draws share positives and readers and are not independent replicates. Refreshing graphs and changing negative draws means the new evaluation score cannot be interpreted as a direct improvement over the old score.

For submission only, refit these same two networks on both the old penultimate-event examples and the refreshed final-event examples. Choose one of the three refreshed negatives per reader using seed418; this balances one earlier and one later example per reader without tripling the later cohort's weight. Selection/final training targets are absent from the underlying features' fitting histories. All these labels are observed in the supplied training CSV. Reuse the accepted fully fitted H component bank for unlabeled test inputs. There is no independent validation score for these production networks after final-event labels join their training set.

No architecture, epochs or blend coefficients are selected using the new final-event outcomes or leaderboard labels. The experiment changes the rolling protocol and expands production meta-training; it does not introduce Item2Vec or a new model family. Retain the previously accepted public prediction file until the new submission is scored.

Run with the original accepted score banks present:

```
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/workspace/work/task2_packages python experiments/task5_rolling.py
```

The runner executes workers sequentially and resumes completed workers from task5_rolling/*.done. Use a fresh output directory if source or input data changes. Factor tables used only as warm starts are removed after their dependent workers complete; refreshed query score banks, fitted production classifiers, margins and results remain. All original datasets and prediction CSVs remain.

## Completed result

Refreshed frozen-stack evaluation accuracy across negative seeds43/441/997:0.9247228816,0.9254027147,0.9244117715 (mean0.9248457893). This measures the graph/protocol refresh with frozen accepted networks, not an independently verified improvement from the production refit.

An additional reader-held-out diagnostic fits the same networks on all86786 earlier selection pairs plus final-event pairs for80% of readers (156214 pairs total), evaluating the remaining17358 readers across three negative draws. Frozen accepted stack averages about92.56635%; refit averages about92.57211%. Mean paired improvement0.00005761 (0.00576 percentage points); approximate reader-clustered95% interval[-0.0008983,0.0010135] includes zero. These cohorts have been reused historically and are diagnostics. The refit has **no convincing local advantage**; public/hidden improvement is unknown.

Production fits on173572 historical pairs and changes1060 of86786 supplied test decisions relative to the accepted public0.9219689812 file. Output SHA256:9deaeea29bc2cf86443b065c6492fcb9f7f2a92a1f2dc4e31e9a63d041d80ad7. All86786 IDs/order/choices, own-target exclusion, zero last-context cold rate, saved-model margin replay and candidate-swap antisymmetry checks pass. Keep the previous accepted prediction CSV until this candidate is scored.

`task5_rolling_sequence.py` is an optional scheduling helper that produces identical fixed sequence-network production/holdout fits while the independent BPR worker runs. `task5_rolling_holdout.py` runs the reader holdout and `task5_rolling_checks.py` replays the final output. Detailed numerical results are in task5_rolling_results.json.
