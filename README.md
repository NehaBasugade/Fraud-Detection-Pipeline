# Fraud Detection: Tabular vs Graph Models Under Strict Temporal Evaluation

## Overview

This project evaluates whether Graph Neural Networks (GNNs) improve fraud detection compared with strong tabular models under a strict, production-realistic temporal setup.

Using the IEEE-CIS Fraud Detection dataset, I built and compared:

- temporal tabular baselines
- a leakage-aware card-only transaction graph
- a standalone GNN baseline
- a minimal hybrid model combining LightGBM with GNN prediction scores

The goal was not to force a graph win. The goal was to test whether graph structure adds real incremental value once evaluation is made realistic and leakage-safe.

---

## Why this project matters

Fraud detection is a high-stakes, highly imbalanced classification problem where weak evaluation choices can create misleading results. Random splits, future-aware features, and weak baselines can make complex models look better than they really are.

This project was designed around a stricter question:

**Does a graph model still help when compared against a strong LightGBM baseline under a chronological split with no future leakage?**

---

## Dataset

- **Dataset:** IEEE-CIS Fraud Detection
- **Total transactions used:** 590,540

---

## Evaluation design

The entire project uses a strict chronological split by `TransactionDT`:

- **Train:** 413,378
- **Validation:** 88,581
- **Test:** 88,581

Fraud prevalence remained stable across splits:

- **Train:** 3.5169%
- **Validation:** 3.4341%
- **Test:** 3.4804%

This setup prevents temporal leakage and better reflects production fraud modeling.

---

## Project phases

### Phase 1 — Temporal data foundation

Built the chronological split and preprocessing pipeline used throughout the project.

### Phase 2 — Tabular baselines

Trained Logistic Regression and LightGBM on 389 engineered tabular features.

**Main benchmark:** LightGBM

**LightGBM test results:**

- **ROC-AUC:** 0.8915
- **PR-AUC:** 0.5151
- **Recall @ Precision >= 0.80:** 0.3477
- **Recall @ Precision >= 0.90:** 0.2248

**Conclusion:**  
LightGBM is the real model to beat.

### Phase 3 — Leakage-aware graph construction

Explored transaction relationships using:

- card
- address
- device

After leakage-safe review, only the **card** relation was retained for mainline modeling.

**Final graph design:**

- transaction nodes linked to card entities
- 14,845 card nodes
- one card edge per transaction
- no missing-value hub nodes
- no future-aware or label-derived graph features

### Phase 4 — Standalone neural / GNN baselines

Built:

- plain MLP baseline
- strict card-only GNN baseline

The GNN beat the plain neural baseline, showing that the graph contains relational signal. However, it was not competitive with LightGBM.

**Standalone GNN test results:**

- **ROC-AUC:** 0.8414
- **PR-AUC:** 0.2602
- **Recall @ Precision >= 0.80:** 0.0415
- **Recall @ Precision >= 0.90:** 0.0000

**Conclusion:**  
The graph signal is real, but not strong enough to justify a graph-first model.

### Phase 5 — Minimal defendable hybrid

Tested whether GNN predictions could add value as a feature to LightGBM.

To make this methodologically valid, I generated leakage-safe out-of-fold train GNN scores using an expanding-window setup before training the hybrid.

**Test results**

**Control (LightGBM):**

- **ROC-AUC:** 0.8968
- **PR-AUC:** 0.5355
- **Recall @ Precision >= 0.80:** 0.3574
- **Recall @ Precision >= 0.90:** 0.2053

**Hybrid (LightGBM + GNN score):**

- **ROC-AUC:** 0.8966
- **PR-AUC:** 0.5293
- **Recall @ Precision >= 0.80:** 0.3558
- **Recall @ Precision >= 0.90:** 0.2251

**Conclusion:**  
The GNN score adds a narrow high-precision signal, but does not improve the tabular model overall.

---

## Final conclusion

This project does not claim that GNNs outperform tabular models for fraud detection on this dataset.

Instead, it shows that:

- strict temporal evaluation matters
- LightGBM is a very strong fraud baseline
- card-based graph structure contains some signal
- standalone GNNs can underperform strong tabular models
- graph complexity should be justified by measurable incremental value

The main value of the project is methodological honesty:

I tested a plausible graph approach under realistic constraints and reported that it did not deliver a broad enough gain to replace or clearly improve the tabular benchmark.

---

## Repository structure

- `data/processed/` — split datasets
- `artifacts/` — machine-consumable outputs, arrays, models, predictions
- `reports/` — human-readable summaries and metrics
- `src/phase5/` — final score-only hybrid pipeline

---

## Key takeaways

- Strong baselines beat fancy architecture when evaluation is realistic
- Graph signal can exist without being practically useful enough
- Honest negative or mixed results can still make a strong ML project

---

## Tech stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- PyTorch
- PyTorch Geometric

---

## Project summary in one line

A fraud detection project that rigorously tests whether graph models add value beyond strong tabular baselines under strict temporal, leakage-safe evaluation.