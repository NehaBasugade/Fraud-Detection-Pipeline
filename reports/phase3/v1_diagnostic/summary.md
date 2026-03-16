# Phase 3 Graph Construction Summary

Graph version: `phase3_v1`

## Schema
- Node types: transaction, card_entity, address_entity, device_entity
- Edge types: transaction_to_card, transaction_to_address, transaction_to_device
- Missing entity values do not create nodes or edges.
- Raw fields used to define entity structure were removed from transaction node features.

## Counts
- Transactions: 590,540
- Card entities: 14,845
- Address entities: 437
- Device entities: 0
- Card edges: 590,540
- Address edges: 524,834
- Device edges: 0

## Connectivity
- Transactions with >=1 entity link: 590,540 (100.00%)
- Transactions with >=2 entity links: 524,834 (88.87%)
- Transactions with all 3 entity links: 0 (0.00%)

## Degree diagnostics
### Card entities
- Mean degree: 39.78
- Median degree: 4.00
- 90th percentile degree: 41.00
- 99th percentile degree: 676.24
- Max degree: 14112
- Singleton rate: 27.48%

### Address entities
- Mean degree: 1200.99
- Median degree: 2.00
- 90th percentile degree: 1588.00
- 99th percentile degree: 25129.48
- Max degree: 46324
- Singleton rate: 39.13%

### Device entities
- Mean degree: 0.00
- Median degree: 0.00
- 90th percentile degree: 0.00
- 99th percentile degree: 0.00
- Max degree: 0
- Singleton rate: 0.00%
