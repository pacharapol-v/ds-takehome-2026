"""Build outputs/report.html: a single self-contained page (Plotly bundled inline, no internet needed).

Run from test-case-1/1_data_analysis:
    python scripts/build_report.py
All numbers come from src/prep.py, the same cleaning the notebooks use.
"""
import html
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import prep  # noqa: E402

OUT = ROOT / "outputs" / "report.html"
AUTHOR = "Pacharapol Vorravanpreecha (Pete)"

# Colour roles (validated reference palette, slots 1-2), light theme.
PA, LIFE, NEUTRAL = "#2a78d6", "#eb6834", "#b4b2a9"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# ---------------------------------------------------------------- data
df, clean_log = prep.load_clean()
raw_rows = len(prep.load_raw())
fin = df[df["has_financials"]]
avg = df["accepted"].mean() * 100


def rates(data, col, order=None, min_n=0):
    r = prep.rates_by(data, col)
    r = r[r["customers"] >= min_n]
    if order is not None:
        r = r.reindex([o for o in order if o in r.index])
    return r


month = rates(df, "campaign_month", prep.MONTHS)
age = rates(df, "age_band", prep.AGE_LABELS)
seg = rates(df, "customer_segment", prep.SEGMENTS)
income = rates(df, "income_band", prep.INCOME_LABELS)
pay = rates(df, "payment_band", prep.PAYMENT_LABELS)
occ = rates(df, "main_occupation", min_n=1000).sort_values("accept_rate")
gender = rates(df, "gender")
# Months on book. MOB 0 is kept as is but labelled: it looks like unfilled fields, not new customers (see data quality).
MOB_LABELS = ["0", "1-3", "4-6", "7-12", "13-24", "25-36", "37-60", "61-120", "121-240", "240+"]
df["mob_band"] = pd.cut(df["mob"], [-0.5, 0.5, 3, 6, 12, 24, 36, 60, 120, 240, np.inf], labels=MOB_LABELS)
mob = rates(df, "mob_band", MOB_LABELS)
mob0 = df["mob"] == 0
mob_new = df["mob"].between(1, 12)
mob0_flat = (df.loc[mob0, "net_flow_30d"] == 0).mean()
mob0_bal_no_in = ((df.loc[mob0, "savacc_bal"] > 0) & (df.loc[mob0, "inflow30d"] == 0)).mean()
cards = pd.concat({
    "Credit card": rates(df, "have_cc").rename({"Y": "has", "N": "none"}),
    "Planet travel card": rates(df, "have_acc_planet").rename({"Y": "has", "N": "none"}),
})
activity = (fin.assign(paid=np.where(fin["easypymt_last_30d"] > 0, "paid", "no payment"))
               .groupby(["account_active", "paid"])
               .agg(customers=("label", "size"), pa_rate=("bought_pa", "mean"), life_rate=("bought_life", "mean")))
activity[["pa_rate", "life_rate"]] *= 100
activity = activity.reindex([("dormant", "no payment"), ("dormant", "paid"), ("active", "no payment"), ("active", "paid")])

skip = (fin["account_active"] == "dormant") & (fin["easypymt_last_30d"] == 0)
pa_lost = 1 - fin.loc[~skip, "bought_pa"].sum() / fin["bought_pa"].sum()
life_lost = 1 - fin.loc[~skip, "bought_life"].sum() / fin["bought_life"].sum()

# ---------------------------------------------------------------- combined drivers (notebook 03 section 5)
# The strongest single factors crossed in plain rate tables. Every pattern used here holds on two random halves of the data.
fx = df[df["has_financials"]]
HEAT_MOB = ["0", "1-12", "13-60", "61-120", "121+"]
fx_mob = pd.cut(fx["mob"], [-0.5, 0.5, 12, 60, 120, np.inf], labels=HEAT_MOB)
pa_heat = fx.groupby([fx_mob, "account_active"], observed=True)["bought_pa"].agg(["mean", "size"]).unstack()
pa_heat_rate = (pa_heat["mean"] * 100)[["dormant", "active"]]
pa_heat_n = pa_heat["size"][["dormant", "active"]]
life_heat = fx.groupby(["age_band", "income_band"], observed=True)["bought_life"].agg(["mean", "size"]).unstack()
life_heat_rate, life_heat_n = life_heat["mean"] * 100, life_heat["size"]

is_active = fx["account_active"] == "active"
new_active = is_active & fx["mob"].between(1, 12)
young_earner = (fx["age"] < 40) & (fx["income"] >= 30_000)
dormant_known = (fx["account_active"] == "dormant") & (fx["mob"] > 0)
mob0_active = (fx["mob"] == 0) & is_active


def seg_stats(mask):
    s = fx[mask]
    return {"customers": len(s), "pct": len(s) / len(fx) * 100, "pa": s["bought_pa"].mean() * 100,
            "life": s["bought_life"].mean() * 100, "pa_share": s["bought_pa"].sum() / fx["bought_pa"].sum() * 100,
            "life_share": s["bought_life"].sum() / fx["bought_life"].sum() * 100}


s_new, s_young, s_dorm = seg_stats(new_active), seg_stats(young_earner), seg_stats(dormant_known)
s_new_male = seg_stats(new_active & (fx["gender"] == "Male"))
s_mid = seg_stats(is_active & fx["mob"].between(13, 120))
pa_avg, life_avg = fx["bought_pa"].mean() * 100, fx["bought_life"].mean() * 100
pa_from_active = fx.loc[is_active, "bought_pa"].sum() / fx["bought_pa"].sum()

# Odds ratios: same model as notebook 02 section 6.
m = df.dropna(subset=["account_active", "age_band", "customer_segment", "income_band", "payment_band", "marital",
                      "scb_payroll", "have_cc", "have_acc_planet", "main_occupation"]).copy()
m["occupation"] = m["main_occupation"].where(m["main_occupation"].isin(
    ["Salary man", "Self-employed", "Freelance", "Student", "Government"]), "Other")
m["paid"] = np.where(m["easypymt_last_30d"] > 0, "Y", "N")
for c in ["age_band", "customer_segment", "income_band"]:
    m[c] = m[c].astype(str)
rhs = ("C(account_active, Treatment('dormant')) + C(paid) + C(age_band, Treatment('30-39'))"
       " + C(customer_segment, Treatment('Mass')) + C(income_band, Treatment('15k-30k')) + C(gender)"
       " + C(marital, Treatment('Single')) + C(occupation, Treatment('Salary man'))"
       " + C(scb_payroll) + C(have_cc) + C(have_acc_planet)")
OR_TERMS = {  # model term -> readable label (a selection; full table in notebook 02)
    "C(account_active, Treatment('dormant'))[T.active]": "Active account (vs dormant)",
    "C(paid)[T.Y]": "Made any payment (vs none)",
    "C(age_band, Treatment('30-39'))[T.23-29]": "Age 23-29 (vs 30-39)",
    "C(age_band, Treatment('30-39'))[T.50-61]": "Age 50-61 (vs 30-39)",
    "C(gender)[T.Male]": "Male (vs female)",
    "C(have_cc)[T.Y]": "Has credit card",
    "C(have_acc_planet)[T.Y]": "Has Planet travel card",
    "C(occupation, Treatment('Salary man'))[T.Student]": "Student (vs salary man)",
    "C(income_band, Treatment('15k-30k'))[T.30k-50k]": "Income 30k-50k (vs 15k-30k)",
    "C(customer_segment, Treatment('Mass'))[T.Lower Mass]": "Lower Mass (vs Mass)",
}
odds = {}
for target in ["bought_pa", "bought_life"]:
    fit = smf.logit(f"{target} ~ {rhs}", data=m).fit(disp=0)
    ci = np.exp(fit.conf_int())
    odds[target] = pd.DataFrame({"or": np.exp(fit.params), "lo": ci[0], "hi": ci[1], "p": fit.pvalues}).loc[list(OR_TERMS)]
    odds[target].index = list(OR_TERMS.values())


# ---------------------------------------------------------------- charts
def base_layout(fig, height=300, ytitle="% of customers offered", legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=INK2),
        hoverlabel=dict(font_size=12), bargap=0.3, bargroupgap=0.08, showlegend=legend,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont_color=MUTED, title=dict(text=ytitle, font_color=MUTED, font_size=11),
                     rangemode="tozero")
    return fig


def grouped_bars(r, height=280):
    x = [str(i) for i in r.index]
    fig = go.Figure()
    for col, name, color in [("pa_rate", "PA", PA), ("life_rate", "Life", LIFE)]:
        fig.add_bar(x=x, y=r[col], name=name, marker_color=color, marker_cornerradius=4,
                    customdata=r["customers"],
                    hovertemplate="%{x}<br>" + name + ": %{y:.2f}%<br>%{customdata:,} customers<extra></extra>")
    return base_layout(fig, height)


def fig_month():
    # Rates on top, volume in its own small panel below: two panels, each with one y-scale.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.08)
    x = [str(i) for i in month.index]
    for col, name, color in [("pa_rate", "PA", PA), ("life_rate", "Life", LIFE)]:
        fig.add_scatter(x=x, y=month[col], name=name, mode="lines+markers", line=dict(color=color, width=2),
                        marker=dict(size=8), hovertemplate="%{x}<br>" + name + ": %{y:.2f}%<extra></extra>", row=1, col=1)
    fig.add_bar(x=x, y=month["customers"], name="Customers offered", marker_color=NEUTRAL, marker_cornerradius=3,
                showlegend=False, hovertemplate="%{x}: %{y:,} offered<extra></extra>", row=2, col=1)
    base_layout(fig, 340)
    fig.update_yaxes(title=dict(text="offered", font_color=MUTED, font_size=11), tickformat="~s", row=2, col=1)
    return fig


def fig_activity():
    labels = ["Dormant,<br>no payment", "Dormant,<br>paid", "Active,<br>no payment", "Active,<br>paid"]
    fig = go.Figure()
    for col, name, color in [("pa_rate", "PA", PA), ("life_rate", "Life", LIFE)]:
        fig.add_bar(x=labels, y=activity[col], name=name, marker_color=color, marker_cornerradius=4,
                    customdata=activity["customers"], text=[f"{v:.2f}%" for v in activity[col]], textposition="outside",
                    textfont=dict(size=11, color=INK2),
                    hovertemplate="%{x}<br>" + name + ": %{y:.2f}%<br>%{customdata:,} customers<extra></extra>")
    return base_layout(fig, 300).update_yaxes(range=[0, activity[["pa_rate", "life_rate"]].max().max() * 1.2])


def fig_payment_hist():
    p = fin["easypymt_last_30d"]
    edges = np.r_[0, np.logspace(0, 6.4, 33)]
    fig = go.Figure()
    for lab, name, color in [(0, "Rejected", NEUTRAL), (1, "PA", PA), (2, "Life", LIFE)]:
        v = p[fin["label"] == lab]
        zero = (v == 0).mean() * 100
        cnt, _ = np.histogram(v[v > 0], edges[1:])
        share = cnt / len(v) * 100
        mids = np.sqrt(edges[1:-1] * edges[2:])
        # Paid customers as a line; the zero-payment share as a separate dot, not joined to the line.
        fig.add_scatter(x=mids, y=share, name=name, mode="lines", legendgroup=name,
                        line=dict(color=color, width=2, shape="spline"),
                        hovertemplate=name + "<br>~%{x:,.0f} baht: %{y:.1f}% of group<extra></extra>")
        fig.add_scatter(x=[0.5], y=[zero], name=name, mode="markers", legendgroup=name, showlegend=False,
                        marker=dict(size=11, color=color, line=dict(color="#fcfcfb", width=2)),
                        hovertemplate=name + "<br>no payment: %{y:.1f}% of group<extra></extra>")
    base_layout(fig, 300, "% of each group")
    fig.update_xaxes(type="log", tickvals=[0.5, 10, 100, 1e3, 1e4, 1e5, 1e6],
                     ticktext=["0", "10", "100", "1k", "10k", "100k", "1M"], title=dict(text="payment in last 30 days (baht)", font_color=MUTED, font_size=11))
    return fig


def fig_odds():
    fig = go.Figure()
    labels = list(OR_TERMS.values())[::-1]
    for key, name, color, shift in [("bought_pa", "PA", PA, 0.17), ("bought_life", "Life", LIFE, -0.17)]:
        t = odds[key].loc[labels]
        y = np.arange(len(labels)) + shift
        sig = t["p"] < 0.05
        fig.add_scatter(
            x=t["or"], y=y, name=name, mode="markers",
            marker=dict(size=10, color=[color if s else "rgba(0,0,0,0)" for s in sig], line=dict(color=color, width=2)),
            error_x=dict(type="data", symmetric=False, array=t["hi"] - t["or"], arrayminus=t["or"] - t["lo"],
                         color=color, thickness=1.5, width=0),
            customdata=np.c_[t["lo"], t["hi"], t["p"]], text=labels,
            hovertemplate="%{text}<br>" + name + " odds ratio %{x:.2f} (95% CI %{customdata[0]:.2f}-%{customdata[1]:.2f})"
                          "<br>p = %{customdata[2]:.3f}<extra></extra>")
    base_layout(fig, 420, "")
    fig.add_vline(x=1, line_color="#c3c2b7", line_width=1)
    fig.update_xaxes(type="log", tickvals=[0.25, 0.5, 1, 2, 4], ticktext=["0.25×", "0.5×", "1×", "2×", "4×"],
                     showgrid=True, gridcolor=GRID, title=dict(text="odds of buying vs the reference group (log scale)", font_color=MUTED, font_size=11))
    fig.update_yaxes(tickvals=np.arange(len(labels)), ticktext=labels, showgrid=False, tickfont_color=INK2, rangemode=None)
    return fig


def fig_cards():
    x = ["No credit card", "Credit card", "No Planet card", "Planet card"]
    rows = [cards.loc[("Credit card", "none")], cards.loc[("Credit card", "has")],
            cards.loc[("Planet travel card", "none")], cards.loc[("Planet travel card", "has")]]
    r = pd.DataFrame(rows, index=x)
    return grouped_bars(r, 280)


# Sequential one-hue ramps (light -> dark) for the heatmaps: blue for PA, orange for life.
BLUE_RAMP = [[0, "#eaf2fd"], [0.35, "#86b6ef"], [0.7, "#2a78d6"], [1, "#104281"]]
ORANGE_RAMP = [[0, "#fdf0ea"], [0.35, "#f5ad8c"], [0.7, "#eb6834"], [1, "#9c3a12"]]


def heatmap(rate, n, x, y, ramp, name, xtitle, ytitle, height=300):
    text = [[f"<b>{r:.2f}%</b><br><span style='font-size:10px'>{c:,}</span>" for r, c in zip(rr, cc)]
            for rr, cc in zip(rate.to_numpy(), n.to_numpy())]
    fig = go.Figure(go.Heatmap(
        z=rate.to_numpy(), x=x, y=y, text=text, texttemplate="%{text}", textfont=dict(size=12),
        colorscale=ramp, zmin=0, xgap=2, ygap=2, showscale=False, customdata=n.to_numpy(),
        hovertemplate=f"{ytitle} %{{y}} · %{{x}}<br>{name} rate: %{{z:.2f}}%<br>%{{customdata:,}} customers<extra></extra>"))
    base_layout(fig, height, ytitle, legend=False)
    fig.update_xaxes(side="top", tickfont_color=INK2, title=dict(text=xtitle, font_color=MUTED, font_size=11), linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont_color=INK2, rangemode=None)
    return fig


def fig_pa_heat():
    return heatmap(pa_heat_rate, pa_heat_n, ["Dormant account", "Active account"],
                   ["0 (suspect)"] + HEAT_MOB[1:], BLUE_RAMP, "PA", "account activity, last 30 days", "months on book", 300)


def fig_life_heat():
    return heatmap(life_heat_rate, life_heat_n, [str(c) for c in life_heat_rate.columns], [str(i) for i in life_heat_rate.index],
                   ORANGE_RAMP, "Life", "monthly income (baht)", "age", 280)


FIGS = {
    "month": fig_month(), "age": grouped_bars(age), "segment": grouped_bars(seg), "income": grouped_bars(income),
    "mob": grouped_bars(mob.rename(index={"0": "0 (suspect)"}), 300), "pay": grouped_bars(pay, 300), "activity": fig_activity(), "payhist": fig_payment_hist(),
    "odds": fig_odds(), "cards": fig_cards(),
    "occ": grouped_bars(occ, 300), "gender": grouped_bars(gender, 260),
    "paheat": fig_pa_heat(), "lifeheat": fig_life_heat(),
}


def chart(key):
    return FIGS[key].to_html(full_html=False, include_plotlyjs=False, div_id=f"fig-{key}",
                             config={"displayModeBar": False, "responsive": True})


def table(r, index_name, cols=("customers", "pa_rate", "life_rate", "accept_rate")):
    names = {"customers": "Customers", "pa_rate": "PA %", "life_rate": "Life %", "accept_rate": "Total %"}
    head = "".join(f"<th>{names[c]}</th>" for c in cols)
    body = ""
    for idx, row in r.iterrows():
        cells = "".join(f"<td>{row[c]:,.0f}</td>" if c == "customers" else f"<td>{row[c]:.2f}</td>" for c in cols)
        label = " / ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        body += f"<tr><th scope='row'>{html.escape(label)}</th>{cells}</tr>"
    return (f"<details><summary>Show data</summary><table><thead><tr><th>{index_name}</th>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></details>")


def heat_table(rate, n, index_name):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in rate.columns)
    body = "".join(f"<tr><th scope='row'>{html.escape(str(i))}</th>"
                   + "".join(f"<td>{rate.loc[i, c]:.2f}% <span class='n'>({n.loc[i, c]:,})</span></td>" for c in rate.columns) + "</tr>"
                   for i in rate.index)
    return (f"<details><summary>Show data</summary><table><thead><tr><th>{index_name}</th>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></details>")


def card(title, sub, key, tbl="", wide=False):
    return (f"<section class='card{' wide' if wide else ''}'><h3>{title}</h3><p class='sub'>{sub}</p>"
            f"{chart(key)}{tbl}</section>")


# ---------------------------------------------------------------- page
n = len(df)
sold = int(df["accepted"].sum())
pa_n, life_n = int(df["bought_pa"].sum()), int(df["bought_life"].sum())
act = fin.groupby("account_active")["bought_pa"].mean() * 100
zero_share = fin["easypymt_last_30d"].eq(0).groupby(fin["label"]).mean()

kpis = [
    ("Customers offered", f"{n:,}", f"{raw_rows - n:,} empty duplicate rows removed"),
    ("Policies sold", f"{sold:,}", f"{sold / n:.2%} acceptance"),
    ("PA policies", f"{pa_n:,}", f"{df['bought_pa'].mean():.2%} rate · {pa_n / sold:.0%} of sales"),
    ("Life policies", f"{life_n:,}", f"{df['bought_life'].mean():.2%} rate · {life_n / sold:.0%} of sales"),
    ("Median payment, 30 days", f"฿{df['easypymt_last_30d'].median():,.0f}", "before the offer"),
    ("Made no payment", f"{(fin['easypymt_last_30d'] == 0).mean():.0%}", "of customers, last 30 days"),
]
kpi_html = "".join(f"<div class='kpi'><div class='kpi-label'>{a}</div><div class='kpi-value'>{b}</div>"
                   f"<div class='kpi-sub'>{c}</div></div>" for a, b, c in kpis)

dq_rows = [
    ("Exact duplicates, mostly null", "1,940", "Dropped: fewer than half of the 26 fields filled, so they add nothing beyond a rejection count."),
    ("Exact duplicates, fully populated", "233", "Kept: no customer ID to prove they are the same person. De-duplicate upstream, before IDs are masked."),
    ("15-day flow columns", "100% of rows", "Identical to the 30-day columns (extraction bug). Dropped."),
    ("currentacc_bal", "100% of rows", "Only takes 0 or 1, cannot be a balance. Dropped."),
    ("Missing financial fields", f"{(~df['has_financials']).sum():,}", "Kept for overall rates; skipped in payment/activity views."),
    ("Months on book longer than adult life", "839", "Set to missing for that field."),
    ("Months on book = 0", f"{mob0.sum():,} ({mob0.mean():.0%})",
     f"Kept as is, but suspect. {mob0_flat:.0%} have net flow of exactly 0 and {mob0_bal_no_in:.0%} hold a balance with no inflow, "
     "which a new account can't do. Mostly in Oct–Feb. And the "
     f"{mob0_active.sum():,} MOB-0 customers with an active account bought no PA at all (about {mob0_active.sum() * act['active'] / 100:.0f} expected). "
     "Likely unfilled fields: excluding them makes the PA activity effect stronger, not weaker."),
    ("Income = 0", "342", "Kept and flagged."),
    ("No premium or policy-status column", "-", "Revenue and premium collection cannot be measured."),
    ("No year or campaign ID", "-", "Month differences reported, not called seasonality."),
]
glance = [
    ("PA is an active-account product", f"{pa_from_active:.0%} of PA buyers had money moving in or out of their account in the 30 days before the offer. "
     f"PA rate {act['active']:.2f}% active vs {act['dormant']:.2f}% dormant."),
    ("Best PA prospects: new and active", f"Active accounts opened 1–12 months ago buy PA at <b>{s_new['pa']:.2f}%</b>, "
     f"{s_new['pa'] / pa_avg:.1f}× the PA average ({s_new_male['pa']:.1f}% for men)."),
    ("Life is a young-earner product", f"Under-40s earning 30k+ baht a month buy life at <b>{s_young['life']:.2f}%</b>, "
     f"{s_young['life'] / life_avg:.1f}× the life average. Dormant customers also lean to life, not PA."),
]
glance_html = "".join(f"<section class='insight'><h3>{a}</h3><p>{b}</p></section>" for a, b in glance)

playbook = [
    ("Active, account opened 1–12 months ago", s_new, "PA first, as part of onboarding", "pa"),
    ("Active, account 13–120 months old", s_mid, "PA", "pa"),
    ("Under 40, income 30k+ baht", s_young, "Life", "life"),
    ("Dormant account (MOB 1+)", s_dorm, "Life only; PA rarely sells here", "life"),
]
playbook_html = "".join(
    f"<tr><th scope='row'>{a}</th><td>{s['customers']:,}</td><td>{s['pct']:.1f}%</td>"
    f"<td class='{'hi' if k == 'pa' else ''}'>{s['pa']:.2f}%</td><td class='{'hi' if k == 'life' else ''}'>{s['life']:.2f}%</td>"
    f"<td>{s['pa_share']:.0f}% / {s['life_share']:.0f}%</td><td>{offer}</td></tr>"
    for a, s, offer, k in playbook)

dq_html = "".join(f"<tr><th scope='row'>{a}</th><td>{b}</td><td>{c}</td></tr>" for a, b, c in dq_rows)

# Light theme only, whatever the viewer's OS setting: every reviewer sees the same colours.
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Campaign Response Report</title>
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --line:rgba(11,11,11,.10); --grid:#e1e0d9; --pa:{PA}; --life:{LIFE}; --accent-bg:#eef4fc; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
.wrap {{ max-width:1240px; margin:0 auto; padding:32px 24px 48px; }}
header h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-.01em; }}
header p {{ margin:0; color:var(--ink2); max-width:860px; }}
.note {{ margin-top:12px; padding:10px 14px; border-left:3px solid var(--life); background:var(--surface); color:var(--ink2); max-width:900px; }}
.kpis {{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin:24px 0 8px; }}
.kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.kpi-label {{ color:var(--ink2); font-size:12px; }}
.kpi-value {{ font-size:26px; font-weight:650; margin:2px 0; }}
.kpi-sub {{ color:var(--muted); font-size:12px; }}
h2 {{ font-size:18px; margin:36px 0 4px; }}
h2 + .lede {{ margin:0 0 14px; color:var(--ink2); max-width:900px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }}
.grid.three {{ grid-template-columns:repeat(3,1fr); }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px 16px 10px; min-width:0; }}
.card.wide, .insight.wide {{ grid-column:1 / -1; }}
.grid > * {{ min-width:0; }}
.card h3 {{ margin:0; font-size:15px; }}
.card .sub {{ margin:2px 0 6px; color:var(--ink2); font-size:13px; }}
.insight {{ background:var(--accent-bg); border-radius:12px; padding:16px 18px; }}
.insight h3 {{ margin:0 0 6px; font-size:15px; }}
.insight ul {{ margin:0; padding-left:18px; }} .insight li {{ margin:4px 0; }}
.big {{ font-size:34px; font-weight:650; line-height:1.1; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:baseline; }}
details {{ margin:4px 0 6px; }} summary {{ cursor:pointer; color:var(--muted); font-size:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:6px; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:right; padding:4px 8px; border-bottom:1px solid var(--grid); }}
th:first-child, td:first-child {{ text-align:left; }}
thead th {{ color:var(--muted); font-weight:500; }}
.dq th, .dq td {{ text-align:left; vertical-align:top; padding:7px 8px; }}
.dq td:nth-child(2) {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
footer {{ margin-top:36px; color:var(--muted); font-size:12px; }}
.glance {{ margin:16px 0 4px; }} .glance .insight p {{ margin:0; color:var(--ink2); }}
.scroll {{ overflow-x:auto; }}
.playbook {{ min-width:640px; font-size:13px; }}
.playbook th, .playbook td {{ padding:7px 8px; vertical-align:top; }}
.playbook td:last-child, .playbook th:last-child {{ text-align:left; }}
.playbook td.hi {{ font-weight:650; color:var(--ink); }}
.n {{ color:var(--muted); }}
.byline {{ margin:0 0 10px !important; color:var(--muted) !important; font-size:13px; }}
.howto {{ margin:10px 0 0 !important; font-size:13px; }}
/* Print / PDF: keep cards whole, drop hover-only hints and the collapsible tables (the HTML keeps them). */
@media print {{
  @page {{ size:A4; margin:10mm; }}
  body {{ background:#fff; -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
  .wrap {{ max-width:none; padding:0; }}
  .card, .insight, .kpi, .playbook tr, .dq tr {{ break-inside:avoid; }}
  h2, h2 + .lede {{ break-after:avoid; }}
  details, .screen-only {{ display:none; }}
  .grid.three.glance {{ grid-template-columns:repeat(3,1fr); }}
}}
@media (max-width:1000px) {{ .kpis {{ grid-template-columns:repeat(3,1fr); }} .grid.three {{ grid-template-columns:1fr 1fr; }} }}
@media (max-width:720px) {{ .wrap {{ padding:20px 16px 40px; }} .kpis {{ grid-template-columns:1fr 1fr; }}
  .grid, .grid.three {{ grid-template-columns:1fr; }} .kpi-value {{ font-size:22px; }} }}
</style>
<script>{plotly.offline.get_plotlyjs()}</script>
</head><body><div class="wrap">

<header>
  <h1>Insurance campaign response: Policy &amp; Payment</h1>
  <p class="byline">{AUTHOR} · {date.today().day} {date.today():%B %Y} · Data: dsc_test_case.csv</p>
  <p>{n:,} bank customers were offered a PA (personal accident) or life insurance policy over 12 campaign months.
  <b>Policy</b> = whether they bought (<span class="swatch" style="background:var(--pa)"></span>PA,
  <span class="swatch" style="background:var(--life)"></span>Life). <b>Payment</b> = what they paid from their bank account in the 30 days before the offer.</p>
  <div class="note">Payment here is <b>pre-offer bank behaviour, not premium</b>. The dataset has no premium or policy-status column, so this page measures who buys, not revenue.
  All rates are % of customers offered.</div>
  <p class="howto"><b>How to read:</b> <span class="swatch" style="background:var(--pa)"></span>PA is blue and
  <span class="swatch" style="background:var(--life)"></span>life is orange throughout.<span class="screen-only"> Hover any chart for exact numbers, or open
  "Show data" under it for the table.</span></p>
</header>

<div class="kpis">{kpi_html}</div>
<div class="grid three glance">{glance_html}</div>

<h2>1. Policy: who buys</h2>
<p class="lede">Only {avg:.2f}% accept. PA outsells life about 3 to 2. Young customers buy both products more often; beyond that, PA and life attract different customers.</p>
<div class="grid">
  {card("By campaign month", "Top: acceptance rate. Bottom: number of customers offered. Jun–Sep had half the volume and the best rates. "
        "Oct–Feb also hold most of the suspect MOB-0 rows, so part of that gap is data, not season.", "month", table(month, "Month"), wide=True)}
  {card("By months on book (MOB)", f"Customers who opened an account 1–12 months ago accept at <b>{df.loc[mob_new, 'accepted'].mean():.2%}</b>, "
        f"{df.loc[mob_new, 'accepted'].mean() * 100 / avg:.1f}× the average, almost all from PA. PA falls steadily with tenure. "
        f"MOB 0 ({mob0.mean():.0%} of customers) looks like missing data, not new customers: see Data quality.", "mob", table(mob, "Months on book"), wide=True)}
  {card("By age", "Under-30s buy most. Life falls steadily with age; over-50s buy life at a third of the under-30 rate.", "age", table(age, "Age"))}
  {card("By monthly income (baht)", "PA peaks at 5k–15k; life is highest from 30k up.", "income", table(income, "Income"))}
</div>
<div class="grid three" style="margin-top:16px">
  {card("By segment", "Lower Mass leans PA, Upper Mass leans life.", "segment", table(seg, "Segment"))}
  {card("By gender", "Men buy PA much more often; life is the same.", "gender", table(gender, "Gender"))}
  {card("By occupation (≥ 1,000 customers)", "Students buy life most; government staff buy least.", "occ", table(occ, "Occupation"))}
</div>

<h2>2. Payment and account activity</h2>
<p class="lede">The clearest signal in the data. Customers who used their account in the last 30 days are far more likely to buy PA. For life, payment size matters more than activity.</p>
<div class="grid">
  {card("By payment in last 30 days (baht)", "No payment → half the average rate. Any payment lifts PA most; life keeps rising with payment size.", "pay", table(pay, "Payment band"))}
  {card("Account activity × payment", f"PA rate is {act['active']:.2f}% for active accounts vs {act['dormant']:.2f}% for dormant ones. Life barely changes.", "activity",
        table(activity, "Activity / payment", ("customers", "pa_rate", "life_rate")))}
  {card("How much each group paid", f"Distribution of 30-day payment. Dots at 0 = share who paid nothing: {zero_share[0]:.0%} of rejecters vs {zero_share[1]:.0%} of PA buyers and {zero_share[2]:.0%} of life buyers.", "payhist", wide=True)}
</div>

<h2>3. What matters once everything else is held constant</h2>
<p class="lede">Logistic regression with all factors together, one model per product. Filled dot = significant at 5%; hollow = not significant. Right of 1× = more likely to buy.</p>
<div class="grid">
  <section class="card"><h3>Odds of buying, by factor</h3><p class="sub">Selected factors, with payment as yes/no. The full model with payment bands is in notebook 02.</p>{chart("odds")}</section>
  <div style="display:grid; gap:16px; align-content:start">
    {card("Card holders swap PA for life", "Credit-card and Planet travel-card holders rarely buy PA, possibly because the card already covers accidents or travel.", "cards", table(cards, "Card"))}
    <section class="insight"><h3>Simple targeting rule</h3>
      <div class="big">−{skip.mean():.0%} offers</div>
      <p style="margin:6px 0 0">Skipping customers who are <b>dormant and made no payment</b> cuts {skip.mean():.0%} of offers but loses only
      <b>{pa_lost:.1%} of PA sales</b>. It would lose {life_lost:.0%} of life sales, so send that group a life-only offer instead.</p>
    </section>
  </div>
</div>

<h2>4. Combining the drivers: who to target</h2>
<p class="lede">Crossing the strongest single factors (account activity with tenure for PA, age with income for life) gives segments that buy at 2–4× the average.
Every pattern below shows up in both of two random halves of the data, so it is not a lucky cut.</p>
<div class="grid">
  {card("PA: account activity × months on book", f"Each cell: PA rate, with the number of customers below it. Active accounts 1–12 months old reach {s_new['pa']:.2f}%; "
        f"dormant accounts stay near 0.1% at every tenure. MOB 0 with an active account: {fx.loc[mob0_active, 'bought_pa'].sum()} PA buyers out of {mob0_active.sum():,}.",
        "paheat", heat_table(pa_heat_rate, pa_heat_n, "Months on book"))}
  {card("Life: age × income", f"Life rate per cell. Under-40s earning 30k+ buy life at {s_young['life']:.2f}%; at every income, life falls with age.",
        "lifeheat", heat_table(life_heat_rate, life_heat_n, "Age"))}
  <section class="insight wide"><h3>Targeting playbook</h3>
    <p style="margin:0 0 8px">Segments can overlap; use them top-down. Rates are % of customers in the segment
    (averages: PA {pa_avg:.2f}%, life {life_avg:.2f}%).</p>
    <div class="scroll"><table class="playbook"><thead><tr><th>Segment</th><th>Customers</th><th>% of base</th><th>PA rate</th><th>Life rate</th>
    <th>Share of PA / life buyers</th><th>Offer</th></tr></thead><tbody>{playbook_html}</tbody></table></div>
  </section>
</div>

<h2>5. Key takeaways</h2>
<div class="grid">
  <section class="insight"><h3>Findings</h3><ul>
    <li><b>PA is an "active banker" product.</b> An active account gives about 4× the odds of buying PA, even with all other factors held constant.</li>
    <li><b>Life is a "young, better-off" product.</b> Under-40s earning 30k+ buy life at {s_young['life']:.2f}%, about 2× the life average; students and bigger payers also lean to life.</li>
    <li><b>New and active is the best PA segment.</b> Active accounts opened 1–12 months ago buy PA at {s_new['pa']:.1f}%, {s_new['pa'] / pa_avg:.0f}× the PA average. Dormant accounts almost never buy PA but buy life above average ({s_dorm['life']:.2f}%).</li>
    <li><b>Card holders choose life over PA.</b></li>
    <li><b>Single factors are modest; combinations are not.</b> One factor alone moves rates about 2×; activity plus tenure reaches about 4× for PA.</li>
  </ul></section>
  <section class="insight"><h3>Recommendations</h3><ul>
    <li>Follow the targeting playbook: PA at onboarding for new active accounts, PA for other active accounts, life for under-40 earners, life only for dormant accounts.</li>
    <li>Pitch life to under-30s and early-career earners; lead with life for card holders.</li>
    <li>Ask the campaign team why Jun–Sep converted better: was it different targeting, or a separate wave?</li>
    <li>Fix the data: check why MOB and account flows are 0 for a quarter of customers; add premium, policy status, campaign ID and year, and a customer ID.</li>
  </ul></section>
</div>

<h2>6. Data quality</h2>
<p class="lede">What was found and what was done. Full checks in notebook 01.</p>
<section class="card"><table class="dq"><thead><tr><th>Issue</th><th>Rows</th><th>Decision</th></tr></thead><tbody>{dq_html}</tbody></table></section>

<footer>Source: dsc_test_case.csv ({raw_rows:,} raw rows → {n:,} after cleaning). Built by scripts/build_report.py from src/prep.py.
Associations, not causes: the offer channel and message are unknown.</footer>
</div>
</body></html>"""

print(pd.concat({k: v["or"].round(2) for k, v in odds.items()}, axis=1).to_string())
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(page, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
