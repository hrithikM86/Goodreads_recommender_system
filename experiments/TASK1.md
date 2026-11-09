# Task 1: rating prediction

Zero ratings are excluded, following the instructor clarification for Assignment 1. They are not treated as one-star ratings. Only supplied Task 1/2 training data is used to fit models. No hidden or public test labels are available.

## Lecture methods reviewed

Slides/recommendation.pdf covers similarity-weighted rating prediction (Jaccard, cosine, Pearson), latent-factor models, user/item biases, regularization, and implicit-feedback extensions. Slides/regression.pdf covers ridge regularization. Experiments implement user/item bias fitting, biased SVD, user/item neighborhoods, learned ridge neighborhood weights, and validation-selected convex blending. Implicit-feedback SVD++ is discussed in the slides but was not run in this round.

## Evaluation

1,834,017 positive-rating rows. Fixed permutation with seed 42: 1,650,615 training, 91,701 model-selection, and 91,701 evaluation rows (90/5/5). Models and blend weights are selected using the selection partition. The evaluation set was inspected across exploratory rounds; it is a local benchmark, not an untouched competition test. Public leaderboard MSE is not directly comparable to these numbers.

Evaluation MSE:
- User/item biases: 0.600648
- Current 16-factor SVD, regularization 0.1: 0.591835
- 32 factors, regularization 0.1: 0.592060
- 64 factors, regularization 0.1: 0.591788
- 32 factors, regularization 0.05: 0.598807
- 32 factors, regularization 0.02, 100 epochs: 0.704232
- 64 factors, regularization 0.02, 100 epochs: 0.680594
- 128 factors, regularization 0.05, 100 epochs: 0.603732
- Item cosine neighborhood, 20 neighbors: 0.567020
- Item Jaccard neighborhood, 40 neighbors: 0.567624
- User/residual-similarity alternatives: worse than item cosine
- Learned ridge neighbor weights: best single setting about 0.5857
- Selected SVD + item-neighborhood ensemble: **0.554263**

The leaderboard reference is 0.503544. This round has not established that score or a top-20 rank.

## Generate submission predictions

From repository root:

```bash
pip install numpy pandas scipy scikit-surprise
python experiments/task1_submission.py
```

This retrains the selected models on every positive-rating training row and writes Assignment 1/assignment1_2026/task_1_predictions.csv, retaining the test order and user/item IDs. Test IDs are used only as prediction inputs. Predictions are clipped to [1,5]. The notebook's Task 1 cell calls the same function. Existing other-task cells are unchanged.

The blend uses 16-, 32-, and 64-factor SVD models (50 epochs, reg_all=0.1) and item-cosine residual neighborhoods with (neighbors, similarity power) of (10,1), (20,2), and (10,2). Weights are stored in task1_submission.py and task1_results.json.

## Reproduce experiments

```bash
python experiments/task1_round1.py
python experiments/task1_round2.py
python experiments/task1_neighbors.py
python experiments/task1_neighbors2.py
python experiments/task1_neighbors_tune.py
python experiments/task1_learned_neighbors.py
python experiments/task1_blend.py
```

Caches and per-row predictions are saved under the ignored experiments/artifacts directory. task1_results.json records this run. Source CSV files remain unchanged.
