# Agent evaluation — LLM `offline:gpt-5-mini`, embeddings `local:lsa-256`

**32/41 passed** (78%) in 2s; median latency 24 ms; corrective retrievals: 5; regenerations: 0

| Category | Pass | Total |
|---|---|---|
| benefit | 5 | 5 |
| comparison | 1 | 1 |
| legal | 1 | 2 |
| negative | 2 | 3 |
| not_in_doc | 0 | 1 |
| overview | 3 | 5 |
| premium | 8 | 9 |
| underwriting | 12 | 15 |

Failures:

- `cheeva-uw-1`: คุ้มชีวา รับประกันอายุตั้งแต่เท่าไหร่ถึงเท่าไหร่
- `cheeva-uw-3`: ถ้าทุนประกันคุ้มชีวาต่ำกว่า 1 ล้าน ต้องแนบสัญญาเพิ่มเติมอะไร
- `cheevee-overview-2`: คุ้มชีวี เอ็กซ์ตร้า คุ้มครองโรคร้ายแรงสูงสุดกี่โรค
- `wl9020-uw-1`: ตลอดชีพ 90/20 รับประกันถึงอายุเท่าไหร่
- `general-legal-2`: เบอร์ Contact Center ของ InsureX คือเบอร์อะไร
- `en-premium-1`: What is the annual premium of Whole Life 99/99 for a 50-year-old man?
- `en-overview-1`: Does Khum Cheevee Extra pay dividends?
- `neg-missing-premium`: คุ้มชีวา 90/10 ผู้หญิงอายุ 25 เบี้ยเท่าไหร่
- `neg-stock`: หุ้น SCB วันนี้ราคาเท่าไหร่
