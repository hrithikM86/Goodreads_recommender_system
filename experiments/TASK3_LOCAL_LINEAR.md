# Task3 local-linear residual smoothing

Existing smoothers used local averages. New features solve weighted ridge regressions of temporal-bias training residual on signed time distance from the query, using exponential kernels. Predicted intercept estimates residual at target time; slopes regularized toward zero. User tau30/90/365 days, intercept/slope penalties5/20; item tau90/365, penalties20/80. Include local constant and local linear predictions, their difference, slope and log weighted support for25 total features. Invalid dates/unknown groups receive zero residual correction. Both sides of a query date may contribute observed training history, as appropriate for Task3 rating prediction.

Histories exclude all original out-of-sample targets; zero ratings excluded. Base checkpoint training-prediction alignment verified with64-bit temporal IDs. Existing275-feature accepted128-factor bank retained. Fresh XGBoost combiner counts/weights selected on original inner validation only. Matched constant control uses same residuals, kernels and support but no slopes. Historical cohorts reused.

Inner MSE: accepted.6793864603; local-linear blend.6789988292, selected642 trees and42.73849% new/57.26151% accepted. Constant control received zero weight. Final local holdout: accepted.6770932158; local-linear blend.6771502712; constant-control blend unchanged.6770932158. Thus inner gain did not transfer to the final holdout; NO accepted-model replacement is recommended from local results. The new CSV is experimental for user-requested checking. Public reference remains.6801266912314015; new public score unknown.

Production features use all provided positive training rows. Fresh combiner refits all140536 known out-of-sample targets. Checks: weighted-ridge solution matches direct solve; large slope penalty recovers local constant; identical timestamps stable; finite features; production model save/reload replay;20000 output rows with original input columns/order and ratings finite in[1,5]; accepted CSV unchanged.

Output: /workspace/output/task3_local_linear/task_3_predictions.csv.
