"""Load and clean the campaign-response dataset (dsc_test_case.csv).

One row = one customer offered an insurance campaign in a given month.
  Policy  = `label`              (0 = rejected, 1 = bought PA insurance, 2 = bought life insurance)
  Payment = `easypymt_last_30d`  (customer's bank payments in the 30 days BEFORE the offer;
                                  it is NOT the premium - the dataset has no premium column)

Every cleaning step is recorded in a log so the report can show exactly what changed and why.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_CSV = DATA_DIR / "dsc_test_case.csv"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEGMENTS = ["Lower Mass", "Mass", "Upper Mass"]
LABELS = {0: "Rejected", 1: "PA (personal accident)", 2: "Life"}

# Thai marital status -> English, then grouped so each bucket is big enough to read.
MARITAL_MAP = {
    "โสด": "Single",
    "สมรส": "Married",
    "สมรสจด ทะเบียน": "Married",          # married, registered
    "สมรสไม่ จดทะเบียน": "Married",       # married, not registered
    "หย่าร้าง": "Divorced / separated / widowed",
    "แยกกันอยู่": "Divorced / separated / widowed",
    "ม่าย": "Divorced / separated / widowed",
    "อื่นๆ": "Other",
}

# Columns dropped, with the reason (verified in notebooks/01_profiling.ipynb).
DROP_COLUMNS = {
    "inflow1_15": "identical to inflow30d in 100% of rows (extraction bug)",
    "outflow1_15": "identical to outflow30d in 100% of rows (extraction bug)",
    "net_flow_15d": "identical to net_flow_30d in 100% of rows (extraction bug)",
    "currentacc_bal": "only takes values 0 or 1; 98.8% zero - no information",
}

AGE_BINS = [0, 29, 39, 49, 200]
AGE_LABELS = ["23-29", "30-39", "40-49", "50-61"]
INCOME_BINS = [-np.inf, 5_000, 15_000, 30_000, 50_000, np.inf]
INCOME_LABELS = ["<5k", "5k-15k", "15k-30k", "30k-50k", "50k+"]
# Payment bands: zero gets its own band because 21% of customers made no payment at all.
PAYMENT_BINS = [-np.inf, 0, 250, 500, 1_000, 3_000, np.inf]
PAYMENT_LABELS = ["0", "1-250", "251-500", "501-1k", "1k-3k", "3k+"]


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean dataframe, cleaning log). Rows are only dropped for exact duplicates."""
    log = []

    def note(step, rows_affected, action):
        log.append({"step": step, "rows_affected": int(rows_affected), "action": action})

    df = raw.copy()

    # 1. Exact duplicates. There is no customer ID, so two identical rows could be two real customers
    #    whose IDs were masked. Only drop a duplicate when it is mostly null (fewer than half of its
    #    fields filled): such rows carry almost no information. Fully populated duplicates are kept.
    dup = df.duplicated()
    mostly_null = df.notna().sum(axis=1) < df.shape[1] / 2
    drop = dup & mostly_null
    note("exact duplicate rows, mostly null", drop.sum(), "dropped (all are label 0)")
    note("exact duplicate rows, fully populated", (dup & ~mostly_null).sum(), "kept (may be distinct customers)")
    df = df[~drop].reset_index(drop=True)

    # 2. Redundant / empty columns.
    df = df.drop(columns=list(DROP_COLUMNS))
    note("redundant columns", 0, "dropped " + ", ".join(DROP_COLUMNS))

    # 3. Missing values come in two blocks: rows with no demographics (also no financials),
    #    and rows with demographics but no financials. Keep them: label is never missing,
    #    so they still count in the overall acceptance rate. Charts by a field skip its NaNs.
    no_demo = df["age"].isna()
    no_fin = df["income"].isna()
    df["has_demographics"] = ~no_demo
    df["has_financials"] = ~no_fin
    note("no demographic fields", no_demo.sum(), "kept; shown as 'Unknown' in demographic splits")
    note("no financial fields", no_fin.sum(), "kept; excluded from financial/payment splits")

    # 4. Impossible values -> NaN for that field only (the rest of the row is kept).
    bad_mob = df["mob"] > (df["age"] - 15) * 12      # account older than the customer's adult life
    df.loc[bad_mob, "mob"] = np.nan
    note("mob longer than customer's adult life", bad_mob.sum(), "mob set to NaN")

    bad_kids = df["num_children"] > 10
    df.loc[bad_kids, "num_children"] = np.nan
    note("num_children > 10", bad_kids.sum(), "set to NaN")

    zero_income = df["income"] == 0
    df["income_zero"] = zero_income
    note("income = 0", zero_income.sum(), "kept, flagged (may be unreported income)")

    # 5. Readable / derived fields.
    df["marital"] = df["marital_sta"].map(MARITAL_MAP)
    df["campaign_month"] = pd.Categorical(df["campaign_month"], MONTHS, ordered=True)
    df["customer_segment"] = pd.Categorical(df["customer_segment"], SEGMENTS, ordered=True)
    df["response"] = df["label"].map(LABELS)
    df["accepted"] = (df["label"] > 0).astype(int)
    df["bought_pa"] = (df["label"] == 1).astype(int)
    df["bought_life"] = (df["label"] == 2).astype(int)
    df["age_band"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS)
    df["income_band"] = pd.cut(df["income"], INCOME_BINS, labels=INCOME_LABELS)
    df["payment_band"] = pd.cut(df["easypymt_last_30d"], PAYMENT_BINS, labels=PAYMENT_LABELS)
    # Any money moved in or out of the savings/current account in the last 30 days.
    active = (df["inflow30d"] > 0) | (df["outflow30d"] > 0)
    df["account_active"] = active.map({True: "active", False: "dormant"}).where(df["has_financials"])
    df["has_children"] = np.where(df["num_children"].isna(), np.nan, (df["num_children"] > 0).astype(float))

    return df, pd.DataFrame(log)


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    return clean(load_raw())


def rates_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customers, PA rate, life rate and total acceptance rate (%) for each value of `col`."""
    g = df.groupby(col, observed=True)
    out = pd.DataFrame({
        "customers": g.size(),
        "pa_rate": g["bought_pa"].mean() * 100,
        "life_rate": g["bought_life"].mean() * 100,
        "accept_rate": g["accepted"].mean() * 100,
    })
    return out.round(3)
