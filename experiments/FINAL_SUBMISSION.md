# Final submission location

The canonical independent executable and selected prediction files are on the **main** branch:

https://github.com/hrithikM86/Goodreads_recommender_system/tree/main/Assignment%201/Submission

Code: https://github.com/hrithikM86/Goodreads_recommender_system/blob/main/Assignment%201/Submission/assignment1.py

This **Experiments** branch stores development scripts, kernels, diagnostics and historical reports under `experiments/`. It contains both accepted and rejected experiments; a historical report is not the final submission selection. Use the main submission directory for grading. Development scripts may expect research caches; the main executable embeds all required source and selected settings, and trains independently from supplied raw CSVs.

Verification of the final package is recorded in `final_submission_verification.json`. No full cold six-task retraining or macOS run was repeated during packaging; all six saved-model replays match the canonical CSVs byte-for-byte, and selected new refits were checked separately.
