# Assignment 1 - Goodreads recommender system

Hrithik Mhatre | TSN: 200478215

## Repository locations

| Contents | Branch and path |
|---|---|
| Standalone executable | main / Assignment 1/Submission/assignment1.py |
| Writeup and six predictions | main / Assignment 1/Submission/ |
| Original supplied data and baseline | main / Assignment 1/assignment1_2026/ |
| Experiment source and reports | Experiments / experiments/ |

Final code:
https://github.com/hrithikM86/Goodreads_recommender_system/blob/main/Assignment%201/Submission/assignment1.py

Submission directory:
https://github.com/hrithikM86/Goodreads_recommender_system/tree/main/Assignment%201/Submission

Experiment branch (case-sensitive name: Experiments):
https://github.com/hrithikM86/Goodreads_recommender_system/tree/Experiments/experiments

## Files for submission

- assignment1.py: independent executable source, with task sections and fixed settings.
- writeup.txt: model tables and a short implementation paragraph for every task.
- task_1_predictions.csv through task_6_predictions.csv: canonical prediction files.
- readme.txt: this guide; Assignment 1/readme.md has identical contents.

The course PDF asks for code, writeup and predictions for at least three tasks;
this package covers all six. Submit predictions to the corresponding leaderboard
slots as required by the course interface. Follow the instructor's code/writeup
upload instructions; this local ZIP conveniently groups all submission files.

## Setup and execution

Install a C++17 compiler. On macOS:

    xcode-select --install
    brew install libomp cmake

Create an environment and install the reference package versions:

    conda create -n goodreads_final python=3.12 -y
    conda activate goodreads_final
    python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 scikit-surprise==1.1.5 implicit==0.7.3 catboost==1.2.10 lightgbm==4.7.0 xgboost==3.4.1

With the repository layout shown above, run from Submission:

    python assignment1.py

Or give an explicit data directory:

    python assignment1.py --data-dir "/path/to/assignment1_2026"

To run selected tasks:

    python assignment1.py --data-dir "/path/to/assignment1_2026" --tasks 2 3 6

By default the six final CSVs are written beside assignment1.py. --output-dir
changes that location; --cache-dir changes the generated model/cache location.
Keep the four supplied training CSVs and six test CSVs in the data directory.
The code discovers a sibling assignment1_2026 folder automatically.

## What the program does

The file embeds readable supporting Python source and native kernels, which are
materialized inside a private .assignment1_cache. It trains the selected models
from raw data and reuses completed stages on later runs. No experiment scripts,
previous predictions, model weights or GitHub access are required to execute it.
No hyperparameter search is run; historical fitting stages remain because the
stacking models require out-of-sample training features. Initial full training
can take several hours; allow at least 20 GB free disk and preferably 16 GB RAM.

Only the supplied data are used. Zero ratings are excluded for Tasks 1/3 and kept
as observed interactions for Tasks 2/4/5/6. Test IDs and timestamps are inputs;
hidden test target values are never used. The supplied next-book ID is an input
for Task 6. Predictions preserve the original test schema and row ordering.

The included prediction CSVs are the exact files selected for submission.
Refitting with another compiler standard library, BLAS or package version can
change numerical details or near-tied candidate choices; the reference is
Python 3.12 on Linux/GCC with the pinned versions above. No claim of bit-for-bit
cross-platform refitting is made. Validation details are in writeup.txt and the
experiment branch. Do not upload generated caches, raw data or compiler outputs.

## Verification scope

All six selected pipelines were replayed from saved checkpoints and reproduced
the packaged CSVs byte-for-byte. A clean raw-data refit of the new 128-factor
component and a clean refit of the selected Task 2 XGBoost combiner also matched
the reference scores exactly. All embedded Python modules and local imports
were checked. A complete cold retraining of all six tasks and a macOS run were
not repeated during this packaging pass.
