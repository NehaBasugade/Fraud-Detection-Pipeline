# Phase 4 Baseline Summary — Card-Only GNN

## Model
Strict card-only GNN baseline using transaction features plus historical card context.

## Graph design
- Node types:
  - transaction
  - card
- Relation:
  - transaction -> card
- Mainline graph used only the card relation from Phase 3.

## Temporal protocol
Strict chronological design was enforced:
- Train batches only saw prior train history.
- Validation inference used train + prior validation history only.
- Test inference used train + validation + prior test history only.
- Target transactions did not influence each other inside the same prediction batch.

## Validation results
- ROC-AUC: 0.8462
- PR-AUC: 0.3357
- Recall @ Precision >= 0.80: 0.1252
- Threshold @ Precision >= 0.80: 0.9704
- Recall @ Precision >= 0.90: 0.0769
- Threshold @ Precision >= 0.90: 0.9932

## Test results
- ROC-AUC: 0.8414
- PR-AUC: 0.2602
- Recall @ Precision >= 0.80: 0.0415
- Threshold @ Precision >= 0.80: 0.9972
- Recall @ Precision >= 0.90: 0.0000
- Threshold @ Precision >= 0.90: 1.0000

## Interpretation
The card-only GNN captures useful relational signal and clearly outperforms the plain MLP baseline.

Test comparison against MLP:
- GNN PR-AUC: 0.2602 vs MLP 0.1359
- GNN Recall @ Precision >= 0.80: 0.0415 vs MLP 0.0000
- GNN Recall @ Precision >= 0.90: 0.0000 vs MLP 0.0000

However, the standalone GNN is still far behind the Phase 2 LightGBM benchmark.

Test comparison against LightGBM:
- GNN PR-AUC: 0.2602 vs LightGBM 0.5151
- GNN Recall @ Precision >= 0.80: 0.0415 vs LightGBM 0.3477
- GNN Recall @ Precision >= 0.90: 0.0000 vs LightGBM 0.2248

## Conclusion
The Phase 4 standalone GNN baseline is valid and leakage-conscious, and it shows that the card graph carries some signal. But the surviving card-only relation is not strong enough for the standalone GNN to compete with the LightGBM benchmark under the strict temporal setup used in this project.