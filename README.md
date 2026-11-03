# InsureX — AI Data Science take-home

Pacharapol Vorravanpreecha

## Start here

| Test case | Open this first | Details |
|---|---|---|
| #1 Data Analysis | [`test-case-1/1_data_analysis/outputs/report.html`](test-case-1/1_data_analysis/outputs/report.html) (any browser, works offline) · [PDF copy](test-case-1/1_data_analysis/outputs/report.pdf) | [`1_data_analysis/README.md`](test-case-1/1_data_analysis/README.md) |
| #1 Data Modeling | [`test-case-1/2_data_model/data_model.pdf`](test-case-1/2_data_model/data_model.pdf) | [`2_data_model/README.md`](test-case-1/2_data_model/README.md) |
| #2 RAG sales agent | [`test-case-2/insurex-sales-agent/README.md`](test-case-2/insurex-sales-agent/README.md) (setup, LangGraph structure, results) | [`docs/TECHNICAL.md`](test-case-2/insurex-sales-agent/docs/TECHNICAL.md) |

## Test Case #1 — Data Analysis

The brief's two datasets, Policy and Payment, are both in `dsc_test_case.csv`:
- **Policy** is the `label` column (rejected / bought PA (personal accident) / bought life).
- **Payment** is `easypymt_last_30d`, meaning bank payments in the 30 days *before* the offer (not premium).

The folder contains:
- the data summary and quality checks;
- the analysis of who buys PA vs life;
- an interactive report.

Headline: an active bank account is the strongest signal for PA. Life buyers are driven by life stage (age, income).

## Test Case #1 — Data Modeling

A table design for monthly agent KPI (key performance indicator) validation:
- pass/fail per agent per month;
- pass/fail streaks;
- contract switches after 3 consecutive passes or fails.

`data_model.pdf` holds the ER (entity-relationship) diagram, the data dictionary, the monthly process and the assumptions.

## Test Case #2 — RAG sales agent (LangChain + LangGraph)

An AI agent for InsureX sales staff and customers. It uses ChromaDB as the vector database.

* **Answers questions** about 5 life-insurance products from the official brochures, in Thai or English. Every answer cites its source page. If the brochures don't contain the answer, the agent says so instead of guessing.
* **Bonus 1, collecting leads over MCP (Model Context Protocol):**
  * it starts when a customer says they are interested;
  * it extracts name, occupation, monthly income and phone as a Pydantic model;
  * it validates them and asks for PDPA (Personal Data Protection Act) consent;
  * it then saves the lead to SQLite through an MCP tool. The LLM never writes to the database itself.
* **Bonus 2, session management:** each customer's conversation is a separate LangGraph thread, stored by a SQLite checkpointer, so it survives a restart. Long chats are summarised to keep prompts short.

**LangGraph workflow:**
* Hybrid retrieval: dense vectors plus two kinds of keyword BM25 search, fused with RRF (Reciprocal Rank Fusion), and filtered by product.
* Two cycles:
  * if the retrieved passages aren't relevant, the agent rewrites the query and searches again;
  * if a number in the answer isn't in the sources, it regenerates the answer, and if it still fails, it answers "not found".
* Every LLM step has a rule-based fallback.

**Results with gpt-5-mini:**
* 40/41 test questions answered correctly. The one miss is a correct answer worded differently from the expected text.
* All 4 out-of-scope questions are refused.
* The right brochure section is in the top 4 search results for every test question.
* 60 automated tests that run without an API key.
* Demo logs (a customer journey from question to saved lead, and parallel sessions with a restart) are in `logs/`.

To run it: follow [`test-case-2/insurex-sales-agent/README.md`](test-case-2/insurex-sales-agent/README.md). You'll need `pip install -r requirements.txt`, an OpenAI or Gemini key in `.env`, and then the build-index and chat commands. Full design notes are in [`docs/TECHNICAL.md`](test-case-2/insurex-sales-agent/docs/TECHNICAL.md).

## Folder layout

```
test-case-1/
  1_data_analysis/
    outputs/      report.html, report.pdf, report.svg (same report), shap_importance.csv
    notebooks/    01 profiling & cleaning · 02 analysis & statistics · 03 models (supporting only)
    src/prep.py   shared loading and cleaning rules
    scripts/      build_report.py (builds the HTML), export_report.py (PDF/SVG copies)
    data/         dsc_test_case.csv and its data definition, as received
    docs/TECHNICAL.md  full data-quality log, statistics, models
    README.md, requirements.txt
  2_data_model/
    data_model.pdf, README.md, er_diagram.png/.svg
test-case-2/
  insurex-sales-agent/      LangGraph RAG agent with MCP lead capture and session memory (full source)
    README.md               setup, how to use, LangGraph structure, results
    docs/TECHNICAL.md       full design notes, evaluation, known issues
    logs/                   demo transcripts · eval/results/ evaluation reports
```

To rerun the analysis, see "How to reproduce" in [`1_data_analysis/README.md`](test-case-1/1_data_analysis/README.md#how-to-reproduce).
