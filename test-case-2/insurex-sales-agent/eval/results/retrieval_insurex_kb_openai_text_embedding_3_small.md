# Retrieval evaluation — embeddings `openai:text-embedding-3-small`

37 answerable questions (Thai + English), top-4 parent sections.

| Config | Hit@1 | Hit@4 | MRR | ProductP@4 | misses |
|---|---|---|---|---|---|
| dense only | 0.62 | 0.89 | 0.73 | 0.86 | cheeva-uw-1, wl9999-premium-1, term-uw-3, en-premium-1 |
| BM25 word only | 0.86 | 1.00 | 0.92 | 0.73 | - |
| BM25 char-3gram only | 0.81 | 0.97 | 0.88 | 0.69 | cheeva-premium-2 |
| hybrid (RRF) | 0.89 | 1.00 | 0.95 | 0.74 | - |
| hybrid + section boosts | 0.92 | 1.00 | 0.96 | 0.74 | - |
| hybrid + product filter | 0.89 | 1.00 | 0.94 | 1.00 | - |
| hybrid + boosts + product filter | 0.92 | 1.00 | 0.95 | 1.00 | - |
