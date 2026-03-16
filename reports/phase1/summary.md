
Phase 1 Summary

The IEEE-CIS fraud training data was processed using a chronological 70/15/15 train/validation/test split based on TransactionDT.
The final split sizes were:
- Train: 413,378
- Validation: 88,581
- Test: 88,581

Temporal ordering was verified successfully:
- Train max TransactionDT = 10437996
- Val min TransactionDT = 10438003
- Val max TransactionDT = 13151840
- Test min TransactionDT = 13151880

Fraud prevalence remained stable across splits:
- Train fraud rate = 3.5169%
- Validation fraud rate = 3.4341%
- Test fraud rate = 3.4804%

Corrected feature typing identified:
- Total usable features = 391
- Numeric features = 377
- Categorical features = 14

The dataset remains highly imbalanced, so future modeling must prioritize PR-AUC and recall-oriented evaluation instead of accuracy.
Missingness analysis shows that several columns have high null rates, and columns with extreme missingness will need deliberate handling in the baseline modeling phase.

Numeric preprocessing artifacts were fit on training data only and saved for reuse, preventing leakage from validation or test data.
