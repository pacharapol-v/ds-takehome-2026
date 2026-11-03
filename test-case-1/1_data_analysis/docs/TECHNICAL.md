# Test Case #1 — Data Analysis: technical notes

> Full data-quality log, findings, models and reproduction steps. For a short overview, see the [README](../README.md).

**Deliverables**
- [`outputs/report.html`](../outputs/report.html): the report. One self-contained interactive file; open it in any browser, no internet needed.
- [`outputs/report.pdf`](../outputs/report.pdf) (A4, printable) and [`outputs/report.svg`](../outputs/report.svg) (the whole page as one image): static copies of the same report.
- [`notebooks/01_profiling.ipynb`](../notebooks/01_profiling.ipynb): data summary, quality checks, every cleaning decision.
- [`notebooks/02_analysis.ipynb`](../notebooks/02_analysis.ipynb): analysis, statistical tests, logistic regression, findings.
- [`notebooks/03_models.ipynb`](../notebooks/03_models.ipynb): binary and 3-class LightGBM models, with SHAP (SHapley Additive exPlanations) explanations. *Supporting exploration only, not used directly in the report or the submission (see [Models](#models-notebook-03)).*

## The data

`data/dsc_test_case.csv`: **215,993 rows × 26 columns**, one row per bank customer offered an insurance campaign in a given month (definitions in `data/data definition.xlsx`).

The brief names two datasets, Policy and Payment. Both are in this one file:

| Brief term | Column | Meaning |
|---|---|---|
| Policy | `label` | 0 = rejected, 1 = bought PA (personal accident), 2 = bought life |
| Payment | `easypymt_last_30d` | what the customer paid from their bank account in the 30 days **before** the offer |

⚠️ **Payment is pre-offer bank behaviour, not premium.** There is no premium, sum-assured or policy-status column, so the analysis measures *who buys*, not revenue.

## Data quality (summary — details in notebook 01)

| Issue | Rows | Decision |
|---|---|---|
| Exact duplicates, mostly null (fewer than 13 of 26 fields filled) | 1,940 | Dropped. They add nothing beyond a rejection count; acceptance moves 1.214% → 1.225% |
| Exact duplicates, fully populated | 233 | Kept. With no customer ID they may be distinct customers (see note below) |
| `inflow1_15`, `outflow1_15`, `net_flow_15d` identical to the 30-day columns | 100% | Dropped (extraction bug) |
| `currentacc_bal` only takes 0 or 1 | 100% | Dropped |
| Financial fields missing as a block | 461 (after de-dup) | Kept for overall rates; skipped in financial splits |
| `mob` (months on book) longer than the customer's adult life | 839 | Set to missing |
| `mob` = 0 | 56,144 (26%) | **Kept as is, but suspect** (see note below) |
| `num_children` > 10 | 2 | Set to missing |
| Income = 0 | 342 | Kept, flagged |
| Campaign month has no year | — | No seasonality claims |
| Definition text for outflow says "Inflow = cash out" | — | Wording error only; data is consistent (net = in − out in 100% of rows) |

Clean data: **214,053 customers**. All cleaning lives in [`src/prep.py`](../src/prep.py), so the notebooks and the report use exactly the same rules.

**Note on duplicates.** The only key in the file is the campaign month, so an identical row is not proof of a double-loaded record. De-duplication belongs **upstream, before customer IDs and personal data are masked**, while a real key still exists.

**Note on MOB (months on book) = 0.** A quarter of customers have MOB 0, but they don't look like new customers:
- 96% have net flow of exactly 0, against 38% of everyone else;
- 86% hold a savings balance with no inflow in the last 30 days, which a newly opened account can't do;
- they come in batches: 61% of October, 49% of February, under 5% of every month from March to September;
- the 2,732 MOB-0 customers with an active account bought **no PA at all**, where about 35 would be expected (chance about 1 in 10^15).

The likely cause is that MOB and the flow fields were never filled for one batch of customers. I kept the rows as they are, so all figures here include them. As a sensitivity check, I excluded them: PA odds for an active account rise from 4.2× to 13.5×, and life tilts slightly towards dormant customers (odds ratio 0.67). So the main finding gets stronger, not weaker.

## Key findings

| | Result |
|---|---|
| Acceptance | **1.22%** overall: 1,634 PA (0.76%), 988 life (0.46%). PA is 62% of policies sold |
| Payment | 21% of customers paid nothing in the last 30 days; they accept at **0.62%**, half the average. Paying customers: 1.2–1.5% |
| Account activity | PA rate **1.27%** with any account inflow/outflow vs **0.26%** without (~5×). Life is flat (0.48% vs 0.44%) |
| Months on book | Customers with an account 1–12 months old accept at **2.08%** (95% CI 1.81–2.39%), 1.7× the average and the best simple segment. It is all PA: 1.58% vs 0.85% for MOB 13+, and 2.0× the PA odds of MOB 13–60 after controls. PA falls steadily with tenure (0.15% past 20 years). Life shows no MOB pattern |
| PA buyer | Active account (odds ratio 4.2), any payment (~2), age 23–29 (1.6), male (1.7), Lower Mass (1.3); much less likely with a credit card (0.44) or Planet travel card (0.30) |
| Life buyer | Age 23–29 (1.4) vs over-50 (0.5), students (1.5), income 30–50k (1.4), payments ≥ 501 baht (~1.5); account activity doesn't matter |
| Combined segments (notebook 03 §5) | **Active accounts opened 1–12 months ago buy PA at 2.98%**, 3.9× the PA average (3.77% for men); dormant accounts stay near 0.1% PA at every tenure. **Under-40s earning 30k+ buy life at 0.96%** (2.1×, 16,824 customers). Dormant accounts buy life at 0.61%, above average. Each pattern holds on two random halves of the data |
| Campaign month | Jun–Sep had half the volume but ~1.8× the acceptance; months with more offers had lower PA rates (Spearman −0.62, p = 0.03). Oct–Feb also hold most MOB-0 rows, so part of the gap is data |
| Targeting rule | Skipping dormant customers with no payment cuts **15% of offers** and loses only **1.8% of PA sales**, but 9% of life sales |

Odds ratios come from a logistic regression with all factors together (notebook 02 §6), so each one is "holding the others constant".
All factors are statistically significant, but each one alone is small (Cramér's V ≤ 0.06; pseudo R² 0.06 for PA, 0.02 for life): a single factor moves rates about 2×. Combining the strongest factors (activity × tenure) reaches about 4× for PA.

## Models (notebook 03)

> **Not used directly in the report or the submission.** The brief asks for data analysis, not a predictive model. I used these models only to check which factors matter most and to pick which ones to cross. The report shows the resulting segments as plain observed rates (notebook 03 §5), with no model output.

Both models are LightGBM (gradient-boosted trees), compared with a logistic-regression baseline. They use the same features and the same 60/20/20 stratified split. `campaign_month` is left out because it has no year, so it can't be used to score a future campaign.

**Imbalance (1.2% buy).** Every model is class-weighted with weight = (majority count) ÷ (class count), from the training rows:
- binary LightGBM: `scale_pos_weight` = 80.6;
- 3-class LightGBM: the same ratio as per-row weights (PA 129.4, life 213.9), because `scale_pos_weight` only exists for binary;
- logistic regression: `class_weight`.

Early stopping uses the validation loss weighted the same way. There is no SMOTE or undersampling and no hyperparameter tuning; the predictive numbers are a sanity check, and the main output is the SHAP explanation.

The models are compared on **PR AUC** (Precision-Recall Area Under the Curve, computed as average precision). A random ranking scores the base rate, so each figure is also shown as "× random".

| Test PR AUC (× random) | A: binary (bought anything) | B: 3-class (rejected / PA / life) |
|---|---|---|
| Any purchase (base 1.22%) | **2.59% (2.1×)** | 2.59% (2.1×), ranking by P(PA)+P(Life) |
| PA (base 0.76%) | - | **2.72% (3.6×)** |
| Life (base 0.46%) | - | **0.98% (2.1×)** |
| Same LightGBM without weighting | 2.67% (2.2×) | PA 2.71% (3.5×) · life 0.94% (2.0×) |
| Class-weighted logistic regression | 2.21% (1.8×) | PA 1.91% (2.5×) · life 0.68% (1.5×) |
| Top 10% of scores | precision 3.2%, recall 26% | PA: precision 2.7%, recall 35% · life: 0.7%, 16% |
| Says which product to offer | No | Yes: in the top 10%, its PA picks buy PA at 3.1% and its life picks buy life at 1.0% |

**What weighting did:**
- PR AUC moved by no more than 0.1 percentage points, because weighting shifts all scores but barely changes their order.
- It cut overfitting: train PR AUC fell from 12.2% to 4.5%, with test at 2.6%.
- It inflated the probabilities (mean P(PA) 0.28 against a real rate of 0.0076), so they are for ranking only.

**I'd use the 3-class model.** It ranks "will buy anything" as well as the binary model, and it also says which product to offer.

What the SHAP values show:
- **PA is driven by account activity.** Zero inflow in the last 30 days is the biggest negative; any inflow is a positive, and the amount barely matters. High savings balance, high income and holding a credit card all lower PA.
- **Life is driven by life-stage:** age first, then tenure, segment, occupation, payment size, marital status and income.
- **Inflow, outflow, income and tenure push PA and life in opposite directions.** These are two different customers.
- **MOB:** the model finds the MOB 1–12 sweet spot for PA on its own, without being given any bands, and PA falls with tenure.

**Limits:**
- Life is hard to predict from these fields.
- The split is random, not by time, because there is no year.
- The probabilities are inflated by the weighting; recalibrate them before reading them as rates.

## Recommendations
1. Offer **PA to customers with an active account**, first of all **at onboarding**: active accounts in their first year buy PA at about 3%. Give dormant customers a life-only offer: they almost never buy PA but buy life above average.
2. Pitch **life to under-40s earning 30k+ baht**, plus students and early-career earners. Lead with life, not PA, for credit-card and travel-card holders.
3. Ask the campaign team why **Jun–Sep converted better**: was it different targeting, or a separate wave?
4. **Fix the data feed**: re-extract the 15-day columns; add premium, policy status, campaign ID and year; de-duplicate before customer IDs are masked.

## Limitations
- **Observational data.** The offer channel and message are unknown. The account-activity effect may partly be channel (e.g. offers shown in the banking app only reach active users).
- **No premium**, so no revenue or collection analysis.
- **No year or campaign ID**, so month differences may be different waves, not seasons.

## How to reproduce

Python 3.11; exact package versions are in [`requirements.txt`](../requirements.txt).

```bash
cd test-case-1/1_data_analysis
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_profiling.ipynb notebooks/02_analysis.ipynb notebooks/03_models.ipynb
python scripts/build_report.py   # independent of the notebooks; reads the CSV through src/prep.py
python scripts/export_report.py  # PDF + SVG copies; needs playwright, pymupdf and a local Chrome
```

## Folder layout
```
data/                      raw CSV + data definition (as received)
src/prep.py                loading, cleaning, derived fields, rate helper
notebooks/01_profiling     data summary + quality checks
notebooks/02_analysis      analysis + model + findings
notebooks/03_models        binary and 3-class LightGBM + SHAP (supporting only)
scripts/build_report.py    builds outputs/report.html
scripts/export_report.py   prints report.html to report.pdf and report.svg
outputs/report.html        the report (interactive)
outputs/report.pdf, .svg   static copies
outputs/shap_importance.csv  SHAP importance per feature (written by notebook 03; not used by the report)
```
