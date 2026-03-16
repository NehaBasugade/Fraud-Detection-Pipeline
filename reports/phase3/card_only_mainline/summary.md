# Phase 3 Card-Only Graph Summary

## Decision
The Phase 3 mainline graph has been finalized as a **card-only** heterogeneous graph with:
- node types: `transaction`, `card_entity`
- edge type: `transaction_to_card`

Address was dropped from the mainline graph because prior diagnostics showed it was hub-dominated and v2 pruning retained only a very small fraction of edges.
Device was not available in the processed parquet data and is not blocking progress.

## Leakage policy
This graph design preserves the project's non-negotiable temporal rules:
- train graph uses train only
- validation inference uses train + prior validation history only
- test inference uses train + validation + prior test history only
- no future-aware aggregates
- no label-derived entity features
- missing values do not create shared hub nodes

## Entity construction
Card entity key is constructed from:
- card1, card2, card3, card5

Columns used for graph structure are removed from transaction node features:
- card1, card2, card3, card5

## Artifact counts
- Train transactions: 413,378
- Validation transactions: 88,581
- Test transactions: 88,581
- Card nodes: 14,845
- Train edges: 413,378
- Validation edges: 88,581
- Test edges: 88,581
- Global edges: 590,540

## Connectivity
- Train transactions with card entity: 413,378 (1.0000)
- Validation transactions with card entity: 88,581 (1.0000)
- Test transactions with card entity: 88,581 (1.0000)
- Global transactions with card entity: 590,540 (1.0000)

## Global card degree diagnostics
- Number of card entities: 14,845
- Number of card edges: 590,540
- Mean degree: 39.78
- Median degree: 4.00
- P90 degree: 41.00
- P99 degree: 676.24
- Max degree: 14112.00
- Singleton entity rate: 0.2748
- Top 1% entity edge share: 0.5297

## Conclusion
The card relation is the only clearly surviving relation in the current processed dataset.
This is the correct graph to carry into the next modeling phase.
Any future attempt to recover device should be treated as a separate improvement branch, not part of the current mainline graph.
