# Phase 4 Baseline Summary — MLP

## Model
Plain transaction-feature MLP baseline trained on Phase 4 prepared features.

## Temporal protocol
- Train used train split only.
- Validation inference used validation transactions in chronological order.
- Test inference used test transactions in chronological order.
- This baseline does not use graph structure.

## Validation results
- ROC-AUC: 0.8143
- PR-AUC: 0.2500
- Recall @ Precision >= 0.80: 0.0631
- Threshold @ Precision >= 0.80: 0.9981
- Recall @ Precision >= 0.90: 0.0000
- Threshold @ Precision >= 0.90: 1.0000

## Test results
- ROC-AUC: 0.8120
- PR-AUC: 0.1359
- Recall @ Precision >= 0.80: 0.0000
- Threshold @ Precision >= 0.80: 1.0000
- Recall @ Precision >= 0.90: 0.0000
- Threshold @ Precision >= 0.90: 1.0000

## Interpretation
The MLP baseline is weak for this task and is not competitive with the Phase 2 LightGBM benchmark.

Compared with LightGBM test performance:
- MLP PR-AUC: 0.1359 vs LightGBM 0.5151
- MLP Recall @ Precision >= 0.80: 0.0000 vs LightGBM 0.3477
- MLP Recall @ Precision >= 0.90: 0.0000 vs LightGBM 0.2248

## Conclusion
A plain neural baseline on the transaction features is not a serious challenger in this project. It serves only as a control baseline for Phase 4 graph experiments.