# Task 5: direct GRU sequence scores added to accepted rolling neural ensemble

Accepted public reference:0.9221302975134238, rank139 as reported by the user. The previous accepted architecture combines224 sequence/preference component scores and260 enhanced-preference scores through separate64/32 MLPs, then averages their antisymmetric probability margins. Those MLPs are score combiners; this experiment adds a model trained directly on ordered book-ID sequences.

## New underlying model

A compact **GRU4Rec-style next-book ranker**, implemented in native C++ with Python orchestration. This is a custom adaptation, not the official GRU4Rec library or an exact reproduction of its session-parallel training implementation. GRUs/RNNs are discussed in the temporal lecture pp118–123. All supplied Task5 reading events count, including rating-zero rows; only user/item columns and CSV order are read.

One24-dimensional GRU layer with separate24-dimensional input and candidate-output book embeddings:

```
z_t = sigmoid(W_z x_t + U_z h_(t-1) + b_z)
r_t = sigmoid(W_r x_t + U_r h_(t-1) + b_r)
n_t = tanh(W_n x_t + U_n (r_t * h_(t-1)) + b_n)
h_t = (1-z_t)*h_(t-1) + z_t*n_t
s(j|history) = h_last dot q_j + bias_j
```

Train on every adjacent next-book transition in each fitting prefix. Mean pairwise BPR softplus loss with four sampled negatives per position. Negatives follow fitting-item frequency^.75 and reject every book in that reader's fitting history. Adagrad learning rate0.03, accumulator initial1e-6, gradient-norm clipping5, weight decay1e-5 on the six GRU matrices;6 epochs, seed2027. Backpropagation truncates every20 positions, carrying the detached hidden state through the remainder of the user's history. Shuffle readers each epoch; keep their internal order. This implementation computes chunk gradients before applying sparse embedding updates; repeated candidate rows receive successive Adagrad updates from those fixed chunk gradients.

Input matrices initialize Normal(0,.2); recurrent matrices orthogonal*.5; update-gate bias-1, other gate biases0; item embeddings Normal(0,.05). Histories start with zero hidden state. No raw user embedding, rating, timestamp, text, external metadata or hidden labels are used. Inference consumes the entire known reader prefix. Untrained input rows and globally cold/unknown candidate scores are zeroed.

## Rolling fits and controlled stacking

Fit three independent GRU models, with the same settings:

- Selection: H[:-2], predict H[-2], using exactly the accepted historical selection negatives.
- Evaluation: H[:-1], predict H[-1] with the corrected three negative draws (43,441,997).
- Submission: all supplied H, score the two supplied test candidates.

Each reader's query target is absent from that GRU's fitting history. Other readers' legitimate earlier interactions remain usable. Three signals (hidden/output dot product, learned candidate bias, their sum) append to both neural banks. No manual old/new-model percentage is introduced. Existing component banks are reused unchanged; the two MLPs are refitted with original64/32 layers, alpha.1, seeds71/73 and16/8 epochs. Existing equal average of the two neural margins remains.

Diagnostic: train neural combiners on all86786 selection examples plus final-event examples from80% of readers, with final labels from17358 readers held out. Compare against the accepted rolling ensemble's cached holdout predictions on those exact readers and three negative draws. The final-event training negative assigned per reader uses seed418, matching the accepted protocol. Three draws share readers/positives; compute approximate uncertainty on per-reader average paired differences. This historical diagnostic has been reused in prior experiments and is not an untouched hidden test.

Submission-only combiners fit on all173572 known historical examples. There is no independent score for these final production weights. Preserve the accepted public prediction file until the new candidate is scored.

## Checks and reproduction

The native recurrent derivative is checked using central finite differences for input/output embeddings, output biases, all GRU gate-matrix families, gate biases and initial hidden state. Independent NumPy equations verify forward hidden states, loss and inference encoder. `task5_gru_checks.py` additionally replays sampled trained hidden states with NumPy, every candidate score from the saved GRU models, prefix event counts, target exclusion, cold/unknown handling, both saved neural classifiers, candidate-swap antisymmetry, all86786 test IDs/order/choices and the output checksum.

```
g++ -std=c++17 -O3 -shared -fPIC experiments/task5_gru.cpp -o /workspace/work/task5_gru.so
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_gru.py --stage gradients
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_gru.py --stage selection
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_gru.py --stage evaluation
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_gru.py --stage full
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_gru.py --stage combine
PYTHONPATH=/workspace/work/task2_packages python experiments/task5_gru_checks.py
```

Selection and evaluation fits may run concurrently. Requires the existing accepted Task5 banks and corrected task5_rolling feature banks. Use a fresh task5_gru output directory if inputs/configuration/source change; cached models/scores assume unchanged inputs. Retain the three GRU model checkpoints and two complete production MLP/scaler checkpoints for replay. Original datasets, accepted models and prediction files are preserved. Only source, documentation and compact reports are published to codex; no datasets, binary models or CSVs are pushed.

## Completed experimental result

The direct GRU standalone evaluation accuracy averages0.7951704960 (79.517%) over the three negative draws. It is an auxiliary signal, not a replacement for the accepted ensemble.

On the matched17358-reader holdout, the GRU-augmented stack averages0.9256634789 versus0.9257210892 for the accepted rolling stack. Paired mean difference-0.00005761 (-0.00576 percentage points); approximate reader-clustered95% interval[-0.00082254,0.00070732] includes zero. There is **no demonstrated local gain**. No hyperparameters or manual blend percentages were selected to improve this holdout.

The production candidate changes657 of86786 supplied test decisions. Treat it as an experimental submission and retain the accepted public0.9221302975 file unless the candidate improves publicly. Public/hidden performance of this new candidate is unknown. Detailed checks/results are in task5_gru_results.json.
