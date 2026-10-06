# Model card

## Intended use
Educational retrospective cancellation-risk analysis and local scenario exploration. Not approved for live booking decisions, automatic cancellation, pricing or overbooking.

## Data and population
119,390 raw records from a city hotel and resort hotel in Portugal, July 2015–August 2017. Exclude 825 zero-guest/zero-night rows. Retain exact duplicates because distinct booking identity is unavailable. Feature values come from historical snapshots near arrival, not verified creation-time records.

## Model and validation
Random forest (120 trees, depth <=14, minimum 20 samples per leaf), selected by validation average precision against dummy, logistic and histogram-boosting models. Sigmoid calibration uses a later disjoint cohort. Review threshold 0.21 uses illustrative FN:FP penalties of 5:1. Calibration and policy tuning share a cohort; final test is untouched.

## Performance
Test ROC AUC 0.766; average precision 0.716; Brier 0.190. At threshold 0.50: 69.0% accuracy, 60.8% precision, 63.8% recall. At 0.21: 59.5% accuracy, 49.8% precision, 91.2% recall. Full results, hotel slices, data hash and interval estimates: [metrics](../reports/metrics.json).

## Limitations
No independent hotel validation; dated population; correlations between repeated/group bookings; unknown creation-time availability; potential drift. No demonstrated treatment effect or realized savings. Country is not used, but other predictors may still encode group differences. No fairness guarantee is established.

## Operational requirements
Timestamped input schema, permissioned data access, authentication, model/version tracking, drift monitoring, prospective human review, and realistic intervention-cost/capacity measurements before deployment. Never load an untrusted joblib file.
