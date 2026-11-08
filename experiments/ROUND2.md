# Task 6 round 2: target MAE 8,457,851

Run from the repository root after installing numpy, pandas, catboost:

```bash
python experiments/task6_rich.py
python experiments/task6_ids.py
```

The first script builds 64 history features and compares MAE and log-target MAE CatBoost models. It uses the same 192,034 training targets and penultimate/final-gap split as round 1. The second adds user and known next-item IDs as categorical inputs. The supplied Task 6 test file contains both IDs, so these features are available at prediction time. Neither script uses hidden labels.

Artifacts are saved under the ignored experiments/artifacts directory. Rating-zero interactions are retained for gap histories; the Assignment 1 zero-rating clarification concerns rating tasks 1 and 3.

These are exploratory local validation experiments. The requested target comes from the public leaderboard and is not directly comparable with this final-gap holdout. Reaching it locally would not guarantee reaching it on the leaderboard, and the converse also applies.

## Results

Final-gap exploratory MAE (seconds):
- Previous ensemble: 9,664,091
- Rich history MAE depth 8: 9,804,008
- Rich history + user/next-item IDs, MAE: 9,526,363
- Rich history + user/next-item IDs, log MAE: 9,456,055
- Validation-selected expanded ensemble: 9,433,913

Expanded blend weights were optimized on penultimate-gap MAE only: 70.39% ID-aware log-target model, 22.08% ID-aware MAE model, 4.96% original RMSE model, 2.57% rich-history MAE model. Tiny numerical weights on other candidates are effectively zero.

The new ensemble improves the prior ensemble by 2.38%, but does not reach the requested 8,457,851 on this local benchmark. Another 10.35% reduction from 9,433,913 would be needed. This target is not directly comparable to a leaderboard score.

To reproduce blending after all candidate runners finish:

```bash
python experiments/task6.py
python experiments/task6_rich.py
python experiments/task6_ids.py
python experiments/task6_blend.py
```

SciPy is additionally required for the constrained blend-weight optimization. Fitted models and caches remain ignored; results and source are committed.
