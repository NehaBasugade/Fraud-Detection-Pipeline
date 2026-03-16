# Phase 3 v2 Graph Construction Summary

Graph version: `phase3_v2`

## Schema
- Node types: transaction, card_entity, address_entity, device_entity
- Edge types: transaction_to_card, transaction_to_address, transaction_to_device
- Missing entity values do not create nodes or edges.
- Address pruning is based on train-only degree thresholds.

## Counts
- Transactions: 590,540
- Card entities: 14,845
- Address entities: 190
- Device entities: 0
- Card edges: 590,540
- Address edges: 4,835
- Device edges: 0

## Device source columns used
- No usable device columns found

## Connectivity
- Transactions with >=1 entity link: 590,540 (100.00%)
- Transactions with >=2 entity links: 4,835 (0.82%)
- Transactions with all entity links: 0 (0.00%)

## Degree diagnostics
### Card entities
- Mean degree: 39.78
- Median degree: 4.00
- 90th percentile degree: 41.00
- 99th percentile degree: 676.24
- Max degree: 14112
- Singleton rate: 27.48%

### Address entities
- Mean degree: 25.45
- Median degree: 5.00
- 90th percentile degree: 41.40
- 99th percentile degree: 430.02
- Max degree: 771
- Singleton rate: 0.00%

### Device entities
- Mean degree: 0.00
- Median degree: 0.00
- 90th percentile degree: 0.00
- 99th percentile degree: 0.00
- Max degree: 0
- Singleton rate: 0.00%
