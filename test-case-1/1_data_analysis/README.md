# Test Case #1 — Data Analysis: Policy & Payment

An analysis of an insurance campaign: **who accepts an insurance offer, and which product they choose**. It covers 214,053 bank customers (after cleaning).

* **The report**: [`outputs/report.html`](outputs/report.html). It is one interactive file that opens in any browser and works offline. There is also a printable [PDF copy](outputs/report.pdf).
* **The data**: both datasets named in the brief are in `data/dsc_test_case.csv`:
  * **Policy** = `label`: rejected, bought PA (personal accident), or bought life.
  * **Payment** = `easypymt_last_30d`: bank payments in the 30 days *before* the offer. This is not premium.
* **Data quality**: I removed duplicates, dropped broken columns and set impossible values to missing. Every rule lives in one file, [`src/prep.py`](src/prep.py), so the notebooks and the report use the same rules.

Full data-quality log, statistics, models and limitations: [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Key findings

* **Acceptance is low: 1.22%.** That is 1,634 PA policies and 988 life policies; PA is 62% of policies sold.
* **PA follows account activity.**
  * Customers whose account had any money in or out in the last 30 days buy PA at 1.27%, against 0.26% for the rest (about 5×).
  * Active accounts opened in the last 12 months buy PA at **2.98%**, 3.9× the PA average.
* **Life follows life stage, not activity.**
  * The best life segment is customers under 40 earning 30k+ baht: **0.96%**, 2.1× the life average.
  * Dormant accounts still buy life slightly above average.
* **A simple targeting rule**: skip dormant customers who made no payment. This cuts 15% of offers but loses only 1.8% of PA sales.

## Recommendations

1. Offer **PA to customers with an active account**, especially at onboarding. Give dormant customers a life-only offer.
2. Pitch **life to under-40s earning 30k+ baht**.
3. Ask the campaign team why **June to September converted better**.
4. **Fix the data feed.** Add premium, policy status, campaign ID and year, and de-duplicate before customer IDs are masked.

## What's in the folder

| Folder | Contents |
|---|---|
| `outputs/` | the report (HTML, PDF, SVG) and SHAP feature importance |
| `notebooks/` | `01` profiling and cleaning · `02` analysis and statistical tests · `03` LightGBM + SHAP (supporting only, not used in the report) |
| `src/prep.py` | shared loading and cleaning rules |
| `scripts/` | `build_report.py` builds the HTML report; `export_report.py` makes the PDF/SVG copies |
| `data/` | the CSV and its data definition, as received |
| `docs/TECHNICAL.md` | full technical notes |

## How to reproduce

Python 3.11.

```bash
cd test-case-1/1_data_analysis
pip install -r requirements.txt
python scripts/build_report.py      # rebuilds outputs/report.html from the CSV
```

To re-run the notebooks or the PDF export, see [docs/TECHNICAL.md](docs/TECHNICAL.md#how-to-reproduce).
