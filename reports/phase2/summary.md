Phase 2 Summary

Tabular baselines were trained on the full IEEE-CIS fraud detection dataset using a strict temporal train/validation/test split.

Feature setup:
- Total modeled features: 389
- Numeric: 375
- Categorical: 14
- Dropped before modeling: TransactionID, TransactionDT, isFraud, D7, dist2

Logistic Regression served as a linear sanity baseline. It reached 0.8448 ROC-AUC and 0.3847 PR-AUC on validation, but performance degraded sharply on the temporally later test split (0.8264 ROC-AUC, 0.1666 PR-AUC). It failed to achieve any threshold with precision >= 0.80 or >= 0.90 on test, indicating poor robustness at operationally useful decision points.

LightGBM provided a strong nonlinear tabular benchmark. It achieved 0.9202 ROC-AUC and 0.5677 PR-AUC on validation, and generalized well to test with 0.8915 ROC-AUC and 0.5151 PR-AUC. Unlike logistic regression, it maintained meaningful high-precision performance on test, reaching 34.8% recall at 80% precision and 22.5% recall at 90% precision.

Conclusion:
LightGBM is the Phase 2 benchmark. Future graph-based models must outperform this baseline on the temporally later test split to justify added graph complexity.
