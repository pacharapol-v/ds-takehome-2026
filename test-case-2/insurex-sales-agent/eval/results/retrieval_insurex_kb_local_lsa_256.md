# Retrieval evaluation — embeddings `local:lsa-256`

37 answerable questions (Thai + English), top-4 parent sections.

| Config | Hit@1 | Hit@4 | MRR | ProductP@4 | misses |
|---|---|---|---|---|---|
| dense only | 0.78 | 1.00 | 0.88 | 0.79 | - |
| BM25 word only | 0.86 | 1.00 | 0.92 | 0.73 | - |
| BM25 char-3gram only | 0.81 | 0.97 | 0.88 | 0.69 | cheeva-premium-2 |
| hybrid (RRF) | 0.95 | 1.00 | 0.97 | 0.71 | - |
| hybrid + section boosts | 0.92 | 1.00 | 0.95 | 0.69 | - |
| hybrid + product filter | 0.95 | 1.00 | 0.97 | 1.00 | - |
| hybrid + boosts + product filter | 0.92 | 1.00 | 0.96 | 1.00 | - |
