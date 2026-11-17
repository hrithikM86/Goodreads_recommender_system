# Task3 multiplicative temporal bias scaling

Extends the existing temporal bias component, rather than duplicating TimeSVD++ latent-factor models. Prediction = global/non-item additive terms + (item bias + item90 bias)*(1 + scale_user + scale_drift*dev(t) + scale_user90 + scale_userday). All additive terms and scale coefficients jointly refitted by exact ridge coordinate updates, initialized from the accepted temporal-bias checkpoint. Scale-deviation penalties20,40,50,100 shrink total scale toward1. Fixed20 passes, no broad hyperparameter search. int64 composite temporal keys prevent overflow.

Three features added to the accepted269-column bank: clipped scaled rating estimate, user-time scale and item effect. All existing components retained. Fresh XGBoost uses the accepted configuration; inner validation selects499 trees and32.59124% augmented/67.40876% accepted predictions before final validation. Scale component fits90% positive training rows only. Zero ratings excluded. Full-data scale refit uses all positive rows, and production combiner uses all140536 historical held-out targets. No external data. Historical validation cohorts reused.

Local MSE: accepted0.6774341252, augmented alone0.6779301735, selected blend0.6771891326. New public MSE pending; accepted public baseline0.6807703420.

Checks: decreasing penalized objective; exact training-prediction replay; zero deviations recover additive model; int32 input IDs handled safely; production scaling-coefficient save/reload replay; XGBoost gzip save/reload replay;20000 output rows; input columns/order preserved; finite ratings in[1,5]; accepted CSV unchanged.

Output: /workspace/output/task3_scaled_temporal/task_3_predictions.csv. The production scaling coefficient archive references immutable indexes/centers in the original full temporal_bias.pkl checkpoint.
